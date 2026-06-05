# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Streaming images and labels from datasets created with dataset_tool.py."""

import logging
import os
import random

import cftime
import cv2
from hydra.utils import to_absolute_path
import numpy as np
import torch
import zarr

from datasets.base import ChannelMetadata, DownscalingDataset
from datasets.img_utils import reshape_fields
from datasets.norm import denormalize, normalize

logger = logging.getLogger(__file__)


# --- Earth-embedding registry ------------------------------------------------
# Regridded satellite embeddings live under <root>/<source>/zarr/ as one static
# annual store per year, on the exact CWA/Europa target grid. Stores are zarr v3
# with an `embedding` array of shape (C, 450, 450). The registry maps a source
# to its filename prefix and channel count; the region maps to a filename token.
_EMB_SOURCES = {  # source -> (filename prefix, channel count)
    "alpha": ("gcs", 64),
    "olmo": ("olmo", 128),
}
_EMB_REGION_TOKEN = {"taiwan": "", "europa": "eu"}
# Array keys tried inside an embedding store, in order. Current per-year stores
# use `embedding`; legacy AlphaEarth stores (e.g. 2024-conservative.zarr) used
# `alpha_earth`. The store root may also be the array itself.
_EMB_ARRAY_KEYS = ("embedding", "alpha_earth")


def _emb_banner(*lines):
    """Log a boxed status banner so the job log makes it unmistakable whether
    (and which) embeddings are active — embeddings must never be silently on/off."""
    bar = "=" * 64
    logger.info(bar)
    for line in lines:
        logger.info(line)
    logger.info(bar)


def _emb_store_path(root, source, region, year, n, masked=False):
    """Build the path to a per-year embedding store, e.g. gcs_2020_N1.zarr.

    When ``masked`` is True the ``_masked`` variant is used, e.g.
    ``olmo_2019_N8_masked.zarr`` (NaN over ocean/no-data, filled with 0 on load)."""
    prefix, _ = _EMB_SOURCES[source]
    tok = _EMB_REGION_TOKEN[region]
    suffix = "_masked" if masked else ""
    name = f"{prefix}{'_' + tok if tok else ''}_{year}_N{n}{suffix}.zarr"
    return os.path.join(root, source, "zarr", name)


def _load_embedding_store(path, img_shape_y, img_shape_x, n=1):
    """Open an embedding zarr store, fill NaNs with 0, crop to the model grid,
    and return a float32 tensor of shape (C, n*img_shape_y, n*img_shape_x).

    ``n`` is the per-axis upsampling factor of the store relative to the weather
    grid: N1 stores are (C, 450, 450) and crop to (C, 448, 448); N8 stores are
    (C, 3600, 3600) and crop to (C, 3584, 3584) = (C, 8*448, 8*448). The extra
    8x8 sub-pixels per weather cell are preserved (NOT averaged) so the model's
    emb branch can learn the intra-cell variation via pixel_unshuffle.

    Accepts both the current per-year stores (array key ``embedding``) and legacy
    AlphaEarth stores (array key ``alpha_earth``, or an array at the store root)."""
    store = zarr.open(path, mode="r")
    if hasattr(store, "shape"):  # opened object is an Array, not a Group
        arr = store
    else:
        arr = next((store[k] for k in _EMB_ARRAY_KEYS if k in store), None)
        if arr is None:
            raise KeyError(
                f"No embedding array found in {path}; tried keys {_EMB_ARRAY_KEYS}."
            )
    emb = arr[:]  # (C, n*450, n*450), float32
    emb = np.where(np.isnan(emb), 0.0, emb)
    emb = emb[:, : n * img_shape_y, : n * img_shape_x].astype(np.float32)
    return torch.as_tensor(emb)


def get_target_normalizations_v1(group):
    """Get target normalizations using center and scale values from the 'group'."""
    return group["cwb_center"][:], group["cwb_scale"][:]


def get_target_normalizations_v2(group):
    """Change the normalizations of the non-gaussian output variables"""
    center = group["cwb_center"]
    scale = group["cwb_scale"]
    variable = group["cwb_variable"]

    center = np.where(variable == "maximum_radar_reflectivity", 25.0, center)
    center = np.where(variable == "eastward_wind_10m", 0.0, center)
    center = np.where(variable == "northward_wind_10m", 0, center)

    scale = np.where(variable == "maximum_radar_reflectivity", 25.0, scale)
    scale = np.where(variable == "eastward_wind_10m", 20.0, scale)
    scale = np.where(variable == "northward_wind_10m", 20.0, scale)
    return center, scale


def get_target_normalizations_europa(group):
    """Europa-tuned normalization (linear rescale, no hardcoded CWA constants).

    Rationale (see normalization_design.md for the full analysis):
      * T2: keep the empirical (center, scale) from the store. Gaussian-ish,
        z-scoring is appropriate.
      * Winds: anchor center at the natural zero (winds are physically
        symmetric in sign; the empirical mean only captures the prevailing
        synoptic flow over the training period). Keep scale at the empirical
        std (~3.4 m/s on Europa). CWA's hardcoded scale=20 m/s is
        typhoon-tuned and would underweight European winds.
      * Precipitation (mm/h): anchor at the natural zero and pick
        scale=5 mm so 5 mm/h -> z=1. This brings the implicit MSE
        loss weight 1/scale^2 from the v1 value of ~2.25 (~152x T2)
        down to 0.04 (~3x T2), removing the zero-inflation artefact
        without applying a non-linear transform. CWA's hardcoded
        scale=25 (chosen for radar dBZ) is not appropriate for mm-rain.
    """
    variable = group["cwb_variable"][:]
    center = np.array(group["cwb_center"][:], dtype=np.float32)
    scale = np.array(group["cwb_scale"][:], dtype=np.float32)

    center = np.where(variable == "eastward_wind_10m", 0.0, center)
    center = np.where(variable == "northward_wind_10m", 0.0, center)
    center = np.where(variable == "precipitation_amount_1hr", 0.0, center)

    scale = np.where(variable == "precipitation_amount_1hr", 5.0, scale)
    return center.astype(np.float32), scale.astype(np.float32)


def get_target_normalizations_v3_europa(group):
    """Europa-tuned normalization with log1p variance stabilization on precip.

    Identical to `europa` for T2 and winds; the precipitation channel is
    additionally pre-transformed with log1p before the linear (center, scale)
    rescale. This converts mm/h to an approximately log-scale quantity --
    conceptually the same operation that dBZ already encodes for CWA radar
    (dBZ = 10 * log10(Z)). After log1p the conditional distribution of
    precip|precip>0 is roughly Gaussian, so z-scoring becomes the right
    operation for it.

    Returned tuple is (center, scale, fwd_transforms, inv_transforms) where
    fwd/inv are per-channel callables (or None for identity). Applied as:
        normalized = (fwd(x) - center) / scale
        denormalized = inv(normalized * scale + center)

    Channel layout on Europa (cwb_variable order):
        0: temperature_2m            -- identity, empirical (mu, sigma)
        1: eastward_wind_10m         -- identity, (0, empirical sigma)
        2: northward_wind_10m        -- identity, (0, empirical sigma)
        3: precipitation_amount_1hr  -- log1p / expm1, (0, 1) post-transform

    Scale=1 post-log1p means a 50 mm/h convective pixel maps to log(51)~3.93,
    comparable to a 3-sigma T2 deviation -- no more single-pixel storms
    dominating the minibatch gradient.

    Caveat: log1p is concave, so back-transforming the regression mean via
    expm1 introduces a small Jensen-style negative bias on the conditional
    mean of precipitation. In the full two-stage CorrDiff pipeline this bias
    is absorbed by the residual diffusion stage. See normalization_design.md
    section 8 for the analysis.
    """
    variable = group["cwb_variable"][:]
    center = np.array(group["cwb_center"][:], dtype=np.float32)
    scale = np.array(group["cwb_scale"][:], dtype=np.float32)

    center = np.where(variable == "eastward_wind_10m", 0.0, center)
    center = np.where(variable == "northward_wind_10m", 0.0, center)
    center = np.where(variable == "precipitation_amount_1hr", 0.0, center)

    # Post-log1p the precip distribution is on a log scale; scale=1 keeps
    # the normalized values in roughly [0, 4] for events up to 50 mm/h.
    scale = np.where(variable == "precipitation_amount_1hr", 1.0, scale)

    n = len(variable)
    fwd = [None] * n
    inv = [None] * n
    precip_idx = int(np.where(variable == "precipitation_amount_1hr")[0][0])
    fwd[precip_idx] = np.log1p
    inv[precip_idx] = np.expm1

    return center.astype(np.float32), scale.astype(np.float32), fwd, inv


def _resolve_target_normalization(result, n_channels):
    """Accept either a 2-tuple (center, scale) or a 4-tuple
    (center, scale, fwd, inv) from a normalization function, and return the
    canonical 4-tuple with identity-by-default transform lists."""
    if len(result) == 2:
        center, scale = result
        return center, scale, [None] * n_channels, [None] * n_channels
    return result


class _ZarrDataset(DownscalingDataset):
    """A Dataset for loading paired training data from a Zarr-file

    This dataset should not be modified to add image processing contributions.
    """

    path: str

    def __init__(
        self, path: str, get_target_normalization=get_target_normalizations_v1
    ):
        self.path = path
        self.group = zarr.open_consolidated(path)
        self.get_target_normalization = get_target_normalization

        # valid indices
        cwb_valid = self.group["cwb_valid"]
        era5_valid = self.group["era5_valid"]
        if not (
            era5_valid.ndim == 2
            and cwb_valid.ndim == 1
            and cwb_valid.shape[0] == era5_valid.shape[0]
        ):
            raise ValueError("Invalid dataset shape")
        era5_all_channels_valid = np.all(era5_valid, axis=-1)
        valid_times = cwb_valid & era5_all_channels_valid
        # need to cast to bool since cwb_valis is stored as an int8 type in zarr.
        self.valid_times = valid_times != 0

        logger.info("Number of valid times: %d", len(self))
        logger.info("input_channels:%s", self.input_channels())
        logger.info("output_channels:%s", self.output_channels())

    def _get_valid_time_index(self, idx):
        time_indexes = np.arange(self.group["time"].size)
        if not self.valid_times.dtype == np.bool_:
            raise ValueError("valid_times must be a boolean array")
        valid_time_indexes = time_indexes[self.valid_times]
        return valid_time_indexes[idx]

    def __getitem__(self, idx):
        idx_to_load = self._get_valid_time_index(idx)
        target = self.group["cwb"][idx_to_load]
        input = self.group["era5"][idx_to_load]

        target = self.normalize_output(target[None, ...])[0]
        input = self.normalize_input(input[None, ...])[0]

        return target, input

    def longitude(self):
        """The longitude. useful for plotting"""
        return self.group["XLONG"]

    def latitude(self):
        """The latitude. useful for plotting"""
        return self.group["XLAT"]

    def _get_channel_meta(self, variable, level):
        if np.isnan(level):
            level = ""
        return ChannelMetadata(name=str(variable), level=str(level))

    def input_channels(self):
        """Metadata for the input channels. A list of dictionaries, one for each channel"""
        variable = self.group["era5_variable"]
        level = self.group["era5_pressure"]
        return [self._get_channel_meta(*v) for v in zip(variable, level)]

    def output_channels(self):
        """Metadata for the output channels. A list of dictionaries, one for each channel"""
        variable = self.group["cwb_variable"]
        level = self.group["cwb_pressure"]
        return [self._get_channel_meta(*v) for v in zip(variable, level)]

    def _read_time(self):
        """The vector of time coordinate has length (self)"""

        return cftime.num2date(
            self.group["time"], units=self.group["time"].attrs["units"]
        )

    def time(self):
        """The vector of time coordinate has length (self)"""
        time = self._read_time()
        return time[self.valid_times].tolist()

    def image_shape(self):
        """Get the shape of the image (same for input and output)."""
        return self.group["cwb"].shape[-2:]

    def _select_norm_channels(self, means, stds, channels):
        if channels is not None:
            means = means[channels]
            stds = stds[channels]
        return (means, stds)

    def _resolved_target_norm(self, channels=None):
        """Return (center, scale, fwd, inv) for the output channels, with
        optional channel subsetting applied uniformly to all four."""
        center, scale, fwd, inv = _resolve_target_normalization(
            self.get_target_normalization(self.group),
            n_channels=int(self.group["cwb_variable"].shape[0]),
        )
        if channels is not None:
            center = np.asarray(center)[channels]
            scale = np.asarray(scale)[channels]
            fwd = [fwd[c] for c in channels]
            inv = [inv[c] for c in channels]
        return center, scale, fwd, inv

    @staticmethod
    def _apply_per_channel(x, fns):
        """Apply per-channel callables (or None for identity) to a (N, C, H, W)
        array along the channel axis. Returns a new array; input is untouched."""
        if all(fn is None for fn in fns):
            return x
        out = np.asarray(x).copy()
        for c, fn in enumerate(fns):
            if fn is not None:
                out[:, c] = fn(out[:, c])
        return out

    def normalize_input(self, x, channels=None):
        """Convert input from physical units to normalized data."""
        norm = self._select_norm_channels(
            self.group["era5_center"], self.group["era5_scale"], channels
        )
        return normalize(x, *norm)

    def denormalize_input(self, x, channels=None):
        """Convert input from normalized data to physical units."""
        norm = self._select_norm_channels(
            self.group["era5_center"], self.group["era5_scale"], channels
        )
        return denormalize(x, *norm)

    def normalize_output(self, x, channels=None):
        """Convert output from physical units to normalized data."""
        center, scale, fwd, _ = self._resolved_target_norm(channels)
        x = self._apply_per_channel(x, fwd)
        return normalize(x, center, scale)

    def denormalize_output(self, x, channels=None):
        """Convert output from normalized data to physical units."""
        center, scale, _, inv = self._resolved_target_norm(channels)
        x = denormalize(x, center, scale)
        return self._apply_per_channel(x, inv)

    def info(self):
        center, scale, _, _ = self._resolved_target_norm()
        return {
            "target_normalization": (center, scale),
            "input_normalization": (
                self.group["era5_center"][:],
                self.group["era5_scale"][:],
            ),
        }

    def __len__(self):
        return self.valid_times.sum()


class FilterTime(DownscalingDataset):
    """Filter a time dependent dataset"""

    def __init__(self, dataset, filter_fn):
        """
        Args:
            filter_fn: if filter_fn(time) is True then return point
        """
        self._dataset = dataset
        self._filter_fn = filter_fn
        self._indices = [i for i, t in enumerate(self._dataset.time()) if filter_fn(t)]

    def longitude(self):
        """Get longitude values from the dataset."""
        return self._dataset.longitude()

    def latitude(self):
        """Get latitude values from the dataset."""
        return self._dataset.latitude()

    def input_channels(self):
        """Metadata for the input channels. A list of dictionaries, one for each channel"""
        return self._dataset.input_channels()

    def output_channels(self):
        """Metadata for the output channels. A list of dictionaries, one for each channel"""
        return self._dataset.output_channels()

    def time(self):
        """Get time values from the dataset."""
        time = self._dataset.time()
        return [time[i] for i in self._indices]

    def info(self):
        """Get information about the dataset."""
        return self._dataset.info()

    def image_shape(self):
        """Get the shape of the image (same for input and output)."""
        return self._dataset.image_shape()

    def normalize_input(self, x, channels=None):
        """Convert input from physical units to normalized data."""
        return self._dataset.normalize_input(x, channels=channels)

    def denormalize_input(self, x, channels=None):
        """Convert input from normalized data to physical units."""
        return self._dataset.denormalize_input(x, channels=channels)

    def normalize_output(self, x, channels=None):
        """Convert output from physical units to normalized data."""
        return self._dataset.normalize_output(x, channels=channels)

    def denormalize_output(self, x, channels=None):
        """Convert output from normalized data to physical units."""
        return self._dataset.denormalize_output(x, channels=channels)

    def __getitem__(self, idx):
        return self._dataset[self._indices[idx]]

    def __len__(self):
        return len(self._indices)


def is_2021(time):
    """Check if the given time is in the year 2021."""
    return time.year == 2021


def is_not_2021(time):
    """Check if the given time is not in the year 2021."""
    return not is_2021(time)


class ZarrDataset(DownscalingDataset):
    """A Dataset for loading paired training data from a Zarr-file with the
    following schema::

        xarray.Dataset {
        dimensions:
                south_north = 450 ;
                west_east = 450 ;
                west_east_stag = 451 ;
                south_north_stag = 451 ;
                time = 8760 ;
                cwb_channel = 20 ;
                era5_channel = 20 ;

        variables:
                float32 XLAT(south_north, west_east) ;
                        XLAT:FieldType = 104 ;
                        XLAT:MemoryOrder = XY  ;
                        XLAT:description = LATITUDE, SOUTH IS NEGATIVE ;
                        XLAT:stagger =  ;
                        XLAT:units = degree_north ;
                float32 XLAT_U(south_north, west_east_stag) ;
                        XLAT_U:FieldType = 104 ;
                        XLAT_U:MemoryOrder = XY  ;
                        XLAT_U:description = LATITUDE, SOUTH IS NEGATIVE ;
                        XLAT_U:stagger = X ;
                        XLAT_U:units = degree_north ;
                float32 XLAT_V(south_north_stag, west_east) ;
                        XLAT_V:FieldType = 104 ;
                        XLAT_V:MemoryOrder = XY  ;
                        XLAT_V:description = LATITUDE, SOUTH IS NEGATIVE ;
                        XLAT_V:stagger = Y ;
                        XLAT_V:units = degree_north ;
                float32 XLONG(south_north, west_east) ;
                        XLONG:FieldType = 104 ;
                        XLONG:MemoryOrder = XY  ;
                        XLONG:description = LONGITUDE, WEST IS NEGATIVE ;
                        XLONG:stagger =  ;
                        XLONG:units = degree_east ;
                float32 XLONG_U(south_north, west_east_stag) ;
                        XLONG_U:FieldType = 104 ;
                        XLONG_U:MemoryOrder = XY  ;
                        XLONG_U:description = LONGITUDE, WEST IS NEGATIVE ;
                        XLONG_U:stagger = X ;
                        XLONG_U:units = degree_east ;
                float32 XLONG_V(south_north_stag, west_east) ;
                        XLONG_V:FieldType = 104 ;
                        XLONG_V:MemoryOrder = XY  ;
                        XLONG_V:description = LONGITUDE, WEST IS NEGATIVE ;
                        XLONG_V:stagger = Y ;
                        XLONG_V:units = degree_east ;
                datetime64[ns] XTIME() ;
                        XTIME:FieldType = 104 ;
                        XTIME:MemoryOrder = 0   ;
                        XTIME:description = minutes since 2022-12-18 13:00:00 ;
                        XTIME:stagger =  ;
                float32 cwb(time, cwb_channel, south_north, west_east) ;
                float32 cwb_center(cwb_channel) ;
                float64 cwb_pressure(cwb_channel) ;
                float32 cwb_scale(cwb_channel) ;
                bool cwb_valid(time) ;
                <U26 cwb_variable(cwb_channel) ;
                float32 era5(time, era5_channel, south_north, west_east) ;
                float32 era5_center(era5_channel) ;
                float64 era5_pressure(era5_channel) ;
                float32 era5_scale(era5_channel) ;
                bool era5_valid(time, era5_channel) ;
                <U19 era5_variable(era5_channel) ;
                datetime64[ns] time(time) ;

    // global attributes:
    }
    """

    path: str

    def __init__(
        self,
        dataset,
        in_channels=(0, 1, 2, 3, 4, 9, 10, 11, 12, 17, 18, 19),
        out_channels=(0, 17, 18, 19),
        img_shape_x=448,
        img_shape_y=448,
        roll=False,
        add_grid=True,
        ds_factor=1,
        train=True,
        all_times=False,
        n_history=0,
        min_path=None,
        max_path=None,
        global_means_path=None,
        global_stds_path=None,
        normalization="v1",
        embedding_path=None,
        embedding_source="none",
        embedding_version="v2_year",
        embedding_n=1,
        embedding_region="taiwan",
        embedding_root=None,
        embedding_static_year=2024,
        embedding_masked=False,
        embedding_separate=None,
    ):
        if not all_times:
            self._dataset = (
                FilterTime(dataset, is_not_2021)
                if train
                else FilterTime(dataset, is_2021)
            )
        else:
            self._dataset = dataset

        self.train = train
        self.img_shape_x = img_shape_x
        self.img_shape_y = img_shape_y
        self.roll = roll
        self.grid = add_grid
        self.ds_factor = ds_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_history = n_history
        self.min_path = min_path
        self.max_path = max_path
        self.global_means_path = (
            to_absolute_path(global_means_path)
            if (global_means_path is not None)
            else None
        )
        self.global_stds_path = (
            to_absolute_path(global_stds_path)
            if (global_stds_path is not None)
            else None
        )
        self.normalization = normalization

        # Load satellite (Alpha Earth / OLMO) embeddings, appended to the model
        # input as extra channels (see __getitem__). The integration style
        # (concat vs emb_branch) is a model-config choice, independent of this
        # loader. There are two ways to select embeddings, in precedence order:
        #
        #   1. embedding_path  -> DIRECT single-file mode (the simple default):
        #      give one zarr file; it is appended to *every* sample (static, all
        #      years). Channel count is read from the file. Revives pre-registry
        #      configs and is the single-year path. Wins over the registry.
        #   2. embedding_source -> the per-year REGISTRY (dormant multi-year option):
        #      embedding_source (none|alpha|olmo) + embedding_version
        #      (v1_static | v2_year) + embedding_root/region/n build the path(s).
        #
        # A banner is always logged so the job output makes it unmistakable
        # whether embeddings are active — they are never silently on/off, and a
        # configured-but-missing file is a hard error, never a silent skip.
        self.embedding_source = embedding_source
        self.embedding_version = embedding_version
        self.embedding_channels = 0
        self.embedding_n = embedding_n  # per-axis upsample factor (1=N1, 8=N8)
        # When True the embedding is returned as a SEPARATE 3rd item from
        # __getitem__ (target, input, embedding) instead of being concatenated
        # onto `input`. Required for N8 (3600x3600) which cannot share the
        # 448x448 weather grid; the emb-branch model receives it via an
        # `embedding=` kwarg and reduces it with pixel_unshuffle. N1 keeps the
        # legacy concat path unchanged.
        #
        # `embedding_n` is the single source of truth: by default the delivery
        # mode is derived from it (n>1 cannot be concatenated, so it must be
        # separate). Pass embedding_separate explicitly only for the rare case of
        # delivering an N1 field as a separate tensor.
        if embedding_separate is None:
            embedding_separate = embedding_n > 1
        self.embedding_separate = bool(embedding_separate)
        self._emb_label = None  # channel-metadata name prefix (set per mode)
        self._year_emb = None  # dict[year -> (C, H, W) tensor] for v2_year
        self._static_emb = None  # (C, H, W) tensor for v1_static / direct path

        if embedding_path is not None:
            # DIRECT single-file mode. Channels named `alpha_earth_*` to stay
            # compatible with checkpoints trained via the old embedding_path knob.
            if embedding_source != "none":
                logger.warning(
                    "Both dataset.embedding_path and dataset.embedding_source are "
                    "set; embedding_path wins, ignoring embedding_source=%s.",
                    embedding_source,
                )
            path = to_absolute_path(embedding_path)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"dataset.embedding_path is set but does not exist: {path}"
                )
            self._static_emb = _load_embedding_store(
                path, img_shape_y, img_shape_x, n=embedding_n
            )
            self.embedding_channels = int(self._static_emb.shape[0])
            self._emb_label = "alpha_earth"
            _emb_banner(
                "EMBEDDINGS: ON  (single file, static for all samples)",
                f"  path     : {path}",
                f"  channels : {self.embedding_channels}  (read from file)",
                f"  grid     : {tuple(self._static_emb.shape)}",
            )
        elif embedding_source == "none":
            _emb_banner(
                "EMBEDDINGS: OFF  (embedding_path unset, embedding_source=none)",
                "  model input = weather channels only (no satellite embeddings)",
            )
        else:
            # REGISTRY mode (dormant multi-year option).
            if embedding_source not in _EMB_SOURCES:
                raise ValueError(
                    f"Unknown embedding_source '{embedding_source}'; "
                    f"expected one of {list(_EMB_SOURCES)} or 'none'"
                )
            if embedding_root is None:
                raise ValueError(
                    "embedding_source is set but dataset.embedding_root is None; "
                    "specify the regrid root in your run config "
                    "(e.g. embedding_root: /home/vault/<group>/<user>/regrid2)"
                )
            self.embedding_channels = _EMB_SOURCES[embedding_source][1]
            self._emb_label = f"{embedding_source}_emb"

            def _path(year):
                return _emb_store_path(
                    embedding_root,
                    embedding_source,
                    embedding_region,
                    year,
                    embedding_n,
                    masked=embedding_masked,
                )

            if embedding_version == "v2_year":
                self._sample_years = [t.year for t in self._dataset.time()]
                self._year_emb = {}
                for year in sorted(set(self._sample_years)):
                    path = _path(year)
                    if not os.path.exists(path):
                        raise FileNotFoundError(
                            f"Year-matched embedding store missing for year {year}: {path}"
                        )
                    self._year_emb[year] = _load_embedding_store(
                        path, img_shape_y, img_shape_x, n=embedding_n
                    )
                _emb_banner(
                    f"EMBEDDINGS: ON  ({embedding_source}, year-matched / v2_year)",
                    f"  channels : {self.embedding_channels}",
                    f"  N factor : {embedding_n}  masked={embedding_masked}  "
                    f"separate={self.embedding_separate}",
                    f"  grid     : {tuple(next(iter(self._year_emb.values())).shape)}",
                    f"  years    : {sorted(self._year_emb)}",
                    f"  root     : {embedding_root}",
                )
            elif embedding_version == "v1_static":
                path = _path(embedding_static_year)
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Static embedding store missing: {path}")
                self._static_emb = _load_embedding_store(
                    path, img_shape_y, img_shape_x, n=embedding_n
                )
                _emb_banner(
                    f"EMBEDDINGS: ON  ({embedding_source}, static / v1_static)",
                    f"  channels : {self.embedding_channels}",
                    f"  N factor : {embedding_n}  masked={embedding_masked}  "
                    f"separate={self.embedding_separate}",
                    f"  year     : {embedding_static_year}  (same field for all samples)",
                    f"  path     : {path}",
                )
            else:
                raise ValueError(
                    f"Unknown embedding_version '{embedding_version}'; "
                    f"expected 'v1_static' or 'v2_year'"
                )

    def info(self):
        """Check if the given time is not in the year 2021."""
        return self._dataset.info()

    def __getitem__(self, idx):
        (target, input) = self._dataset[idx]
        # crop and downsamples
        # rolling
        if self.train and self.roll:
            x_roll = random.randint(0, self.img_shape_x)
        else:
            x_roll = 0

        # channels
        input = input[self.in_channels, :, :]
        target = target[self.out_channels, :, :]

        if self.ds_factor > 1:
            target = self._create_lowres_(target, factor=self.ds_factor)

        reshape_args = (
            x_roll,
            self.train,
            self.n_history,
            self.in_channels,
            self.out_channels,
            self.img_shape_x,
            self.img_shape_y,
            self.min_path,
            self.max_path,
            self.global_means_path,
            self.global_stds_path,
            self.normalization,
            self.roll,
        )
        # SR
        input = reshape_fields(
            input,
            "inp",
            *reshape_args,
            normalize=False,
        )  # 3x720x1440
        target = reshape_fields(
            target, "tar", *reshape_args, normalize=False
        )  # 3x720x1440

        # Select this sample's embedding field (year-matched or static), if any.
        emb = None
        if self._year_emb is not None:
            emb = self._year_emb[self._sample_years[idx]]
        elif self._static_emb is not None:
            emb = self._static_emb

        if emb is not None:
            if self.embedding_separate:
                # SEPARATE path (e.g. N8): keep `input` weather-only and hand the
                # embedding back as a 3rd item; the emb-branch model receives it
                # via an `embedding=` kwarg (pixel_unshuffle front-end).
                return target, input, emb
            # LEGACY concat path (N1): append embedding as extra input channels;
            # the model splits the trailing channels back off (unchanged).
            input = torch.cat([input, emb], dim=0)

        return target, input

    def input_channels(self):
        """Metadata for the input channels. A list of dictionaries, one for each channel.

        In the legacy concat path the embedding rides inside ``input``, so its
        channels are listed here (and counted into the model's input channels).
        In ``embedding_separate`` mode the embedding is delivered as its own
        tensor (and reduced inside the model's branch), so it is NOT part of the
        weather input and must be excluded from this count."""
        in_channels = self._dataset.input_channels()
        in_channels = [in_channels[i] for i in self.in_channels]
        if self.embedding_channels > 0 and not self.embedding_separate:
            emb_channels = [
                ChannelMetadata(name=f"{self._emb_label}_{i}", level="")
                for i in range(self.embedding_channels)
            ]
            in_channels = in_channels + emb_channels
        return in_channels

    def output_channels(self):
        """Metadata for the output channels. A list of dictionaries, one for each channel"""
        out_channels = self._dataset.output_channels()
        return [out_channels[i] for i in self.out_channels]

    def __len__(self):
        return len(self._dataset)

    def longitude(self):
        """Get longitude values from the dataset."""
        lon = self._dataset.longitude()
        return lon if self.train else lon[..., : self.img_shape_y, : self.img_shape_x]

    def latitude(self):
        """Get latitude values from the dataset."""
        lat = self._dataset.latitude()
        return lat if self.train else lat[..., : self.img_shape_y, : self.img_shape_x]

    def time(self):
        """Get time values from the dataset."""
        return self._dataset.time()

    def image_shape(self):
        """Get the shape of the image (same for input and output)."""
        return (self.img_shape_y, self.img_shape_x)

    def normalize_input(self, x):
        """Convert input from physical units to normalized data."""
        x_norm = self._dataset.normalize_input(
            x[:, : len(self.in_channels)], channels=self.in_channels
        )
        return np.concatenate((x_norm, x[:, self.in_channels :]), axis=1)

    def denormalize_input(self, x):
        """Convert input from normalized data to physical units."""
        x_denorm = self._dataset.denormalize_input(
            x[:, : len(self.in_channels)], channels=self.in_channels
        )
        return np.concatenate((x_denorm, x[:, len(self.in_channels) :]), axis=1)

    def normalize_output(self, x):
        """Convert output from physical units to normalized data."""
        return self._dataset.normalize_output(x, channels=self.out_channels)

    def denormalize_output(self, x):
        """Convert output from normalized data to physical units."""
        return self._dataset.denormalize_output(x, channels=self.out_channels)

    def _create_highres_(self, x, shape):
        # downsample the high res imag
        x = x.transpose(1, 2, 0)
        # upsample with bicubic interpolation to bring the image to the nominal size
        x = cv2.resize(
            x, (shape[0], shape[1]), interpolation=cv2.INTER_CUBIC
        )  # 32x32x3
        x = x.transpose(2, 0, 1)  # 3x32x32
        return x

    def _create_lowres_(self, x, factor=4):
        # downsample the high res imag
        x = x.transpose(1, 2, 0)
        x = x[::factor, ::factor, :]  # 8x8x3  #subsample
        # upsample with bicubic interpolation to bring the image to the nominal size
        x = cv2.resize(
            x, (x.shape[1] * factor, x.shape[0] * factor), interpolation=cv2.INTER_CUBIC
        )  # 32x32x3
        x = x.transpose(2, 0, 1)  # 3x32x32
        return x


def get_zarr_dataset(
    *, data_path, normalization="v1", all_times=False, include_times=None,
    embedding_path=None, **kwargs,
):
    """Get a Zarr dataset for training or evaluation.

    If `include_times` is set (list of ISO 8601 strings, e.g. "2021-09-12T00:00:00"),
    the dataset is filtered to only those timestamps and the year-2021 split (is_2021 /
    is_not_2021) is bypassed.

    `embedding_path` (when set) is the simple single-file embedding interface: that
    one zarr file is appended to every sample. It takes precedence over the per-year
    `embedding_source`/`embedding_version` registry (see ZarrDataset.__init__).
    """
    data_path = to_absolute_path(data_path)
    get_target_normalization = {
        "v1": get_target_normalizations_v1,
        "v2": get_target_normalizations_v2,
        "europa": get_target_normalizations_europa,
        "v3_europa": get_target_normalizations_v3_europa,
    }[normalization]
    logger.info(f"Normalization: {normalization}")
    zdataset = _ZarrDataset(
        data_path, get_target_normalization=get_target_normalization
    )
    if include_times is not None:
        include_set = set(include_times)
        def _in_set(t):
            iso = f"{t.year:04d}-{t.month:02d}-{t.day:02d}T{t.hour:02d}:{t.minute:02d}:{t.second:02d}"
            return iso in include_set
        zdataset = FilterTime(zdataset, _in_set)
        all_times = True
        logger.info("include_times: filtered to %d timestamps", len(zdataset))
    return ZarrDataset(
        dataset=zdataset, normalization=normalization, all_times=all_times,
        embedding_path=embedding_path, **kwargs
    )
