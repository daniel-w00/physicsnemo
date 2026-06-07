"""NetCDF loading and the streaming tensor adapter for the evaluation pipeline.

A generated CorrDiff output file has three groups:
    root        — coordinates (lat, lon, time)
    truth       — ground truth,  dims (time, y, x)        per variable
    prediction  — model output,  dims (ensemble, time, y, x) per variable

``GenerationFile`` opens such a file and exposes a single **streaming pass**
(:meth:`iter_timesteps`) that yields one timestep at a time as torch tensors shaped
``(N_ens, C, H, W)`` (prediction) and ``(C, H, W)`` (target). Streaming keeps peak
memory bounded — only one timestep of all channels is resident at once — and lets a
caller update every accumulator (metrics, histogram, RAPSD) in a single read of the file.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Iterator

import numpy as np
import torch
import xarray as xr

# ── Channel definitions ──────────────────────────────────────────────────────
# The four stored output variables plus a derived wind-speed channel. ``signed``
# flags variables that take negative values (wind components) so plots use a
# symmetric colormap. ``derived`` channels are computed from stored ones.


@dataclasses.dataclass(frozen=True)
class Channel:
    name: str
    label: str
    unit: str
    signed: bool = False
    derived: bool = False

    @property
    def diagnostic_info(self) -> dict:
        """Axis label/unit dict consumed by core.plots.plot_diagnostic_panel."""
        return {"label": self.label, "unit": self.unit}


STORED_VARS = (
    Channel("maximum_radar_reflectivity", "Radar reflectivity", "dBZ"),
    Channel("temperature_2m", "2m temperature", "K"),
    Channel("eastward_wind_10m", "10m eastward wind", "m/s", signed=True),
    Channel("northward_wind_10m", "10m northward wind", "m/s", signed=True),
)
WIND_SPEED = Channel("wind_speed_10m", "10m wind speed", "m/s", derived=True)

DEFAULT_CHANNELS = STORED_VARS + (WIND_SPEED,)


def channels_by_name(names: list[str] | None) -> list[Channel]:
    """Resolve a list of channel names to Channel objects (default: all)."""
    if not names:
        return list(DEFAULT_CHANNELS)
    lookup = {c.name: c for c in DEFAULT_CHANNELS}
    missing = [n for n in names if n not in lookup]
    if missing:
        raise ValueError(
            f"Unknown channel(s): {missing}. Available: {sorted(lookup)}"
        )
    return [lookup[n] for n in names]


# ── File handle ──────────────────────────────────────────────────────────────


class GenerationFile:
    """Lazy handle to a generation ``.nc`` file with a streaming timestep iterator."""

    def __init__(self, path: str, channels: list[Channel] | None = None):
        self.path = path
        self.channels = channels or list(DEFAULT_CHANNELS)

        self.root = xr.open_dataset(path)
        self.pred = xr.open_dataset(path, group="prediction").merge(self.root)
        self.truth = xr.open_dataset(path, group="truth").merge(self.root)
        self.pred = self.pred.set_coords(["lon", "lat"])
        self.truth = self.truth.set_coords(["lon", "lat"])

        self.n_time = int(self.truth.sizes["time"])
        self.n_ensemble = int(self.pred.sizes["ensemble"]) if "ensemble" in self.pred.dims else 1
        self.times = self.root["time"].values

        lat = self.root["lat"].values
        lon = self.root["lon"].values
        if lat.ndim == 1 and lon.ndim == 1:
            self.lon2d, self.lat2d = np.meshgrid(lon, lat)
        else:
            self.lat2d, self.lon2d = lat, lon
        self.img_shape = self.lat2d.shape  # (H, W)

    # -- properties ------------------------------------------------------------

    @property
    def stem(self) -> str:
        return os.path.splitext(os.path.basename(self.path))[0]

    @property
    def is_ensemble(self) -> bool:
        return self.n_ensemble > 1

    def dx_km(self) -> float:
        """Mean zonal grid spacing in km, from the lat/lon grid."""
        lon, lat = self.lon2d, self.lat2d
        dlon = float(np.abs(np.diff(lon[lon.shape[0] // 2, :])).mean())
        lat_c = float(lat[lat.shape[0] // 2, lat.shape[1] // 2])
        return dlon * 111.0 * np.cos(np.radians(lat_c))

    # -- per-variable / per-timestep extraction --------------------------------

    def _truth_field(self, ch: Channel, t: int) -> np.ndarray:
        """Ground-truth field for one channel at time index t → (H, W)."""
        if ch.derived:  # wind_speed_10m
            u = self.truth["eastward_wind_10m"].isel(time=t).values
            v = self.truth["northward_wind_10m"].isel(time=t).values
            return np.sqrt(u ** 2 + v ** 2)
        return self.truth[ch.name].isel(time=t).values

    def _pred_field(self, ch: Channel, t: int) -> np.ndarray:
        """Prediction field for one channel at time index t → (N_ens, H, W)."""
        if ch.derived:
            u = self._ens_array("eastward_wind_10m", t)
            v = self._ens_array("northward_wind_10m", t)
            return np.sqrt(u ** 2 + v ** 2)
        return self._ens_array(ch.name, t)

    def _ens_array(self, var: str, t: int) -> np.ndarray:
        arr = self.pred[var].isel(time=t).values
        if "ensemble" not in self.pred[var].dims:
            arr = arr[np.newaxis, ...]  # (1, H, W)
        return arr

    def order(self) -> np.ndarray:
        """Chronological ordering of the (possibly unsorted) time axis."""
        return np.argsort(self.times)

    def iter_timesteps(
        self, channels: list[Channel] | None = None
    ) -> Iterator[tuple[int, np.datetime64, torch.Tensor, torch.Tensor]]:
        """Yield ``(t, time_value, pred_ens, target)`` one timestep at a time.

        ``pred_ens`` is ``(N_ens, C, H, W)`` and ``target`` is ``(C, H, W)``, both
        float32 torch tensors with channels in the order of ``channels``.
        """
        chans = channels or self.channels
        for t in range(self.n_time):
            target = np.stack([self._truth_field(c, t) for c in chans], axis=0)
            pred = np.stack([self._pred_field(c, t) for c in chans], axis=1)  # (N_ens, C, H, W)
            yield (
                t,
                self.times[t],
                torch.from_numpy(np.ascontiguousarray(pred, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(target, dtype=np.float32)),
            )

    def event_fields(self, t: int, channels: list[Channel] | None = None):
        """Full (all-ensemble) fields at one timestep for event-panel plotting.

        Returns ``(pred_ens (N_ens, C, H, W), target (C, H, W))`` as numpy arrays.
        """
        chans = channels or self.channels
        target = np.stack([self._truth_field(c, t) for c in chans], axis=0)
        pred = np.stack([self._pred_field(c, t) for c in chans], axis=1)
        return pred.astype(np.float32), target.astype(np.float32)

    def close(self):
        for ds in (self.root, self.pred, self.truth):
            try:
                ds.close()
            except Exception:
                pass
