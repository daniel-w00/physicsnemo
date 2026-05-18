"""Shared utilities for generation output analysis."""

import dataclasses
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


# Variables in truth/prediction groups
OUTPUT_VARS = [
    "maximum_radar_reflectivity",
    "temperature_2m",
    "eastward_wind_10m",
    "northward_wind_10m",
]

ALL_VARS = OUTPUT_VARS + ["wind_speed_10m"]

VAR_LABELS = {
    "maximum_radar_reflectivity": "Reflectivity [dBZ]",
    "temperature_2m": "T2m [K]",
    "eastward_wind_10m": "U-wind 10m [m/s]",
    "northward_wind_10m": "V-wind 10m [m/s]",
    "wind_speed_10m": "Wind Speed 10m [m/s]",
}

VAR_CMAP = {
    "maximum_radar_reflectivity": "magma",
    "temperature_2m": "RdYlBu_r",
    "eastward_wind_10m": "RdBu_r",
    "northward_wind_10m": "RdBu_r",
    "wind_speed_10m": "viridis",
}

# Fixed color palette for the first few models; falls back to tab20 for larger sets
_PALETTE_FIXED = ["#e07b39", "#3b7dd8", "#2ca02c", "#d62728", "#9467bd",
                  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def _get_palette(n: int) -> list:
    """Return n distinct colors."""
    if n <= len(_PALETTE_FIXED):
        return _PALETTE_FIXED[:n]
    cmap = plt.get_cmap("tab20")
    return [cmap(i / n) for i in range(n)]

# Base results directory (relative to repo root)
RESULTS_BASE = os.path.join(os.path.dirname(__file__), "results")


# ─── Model specification ──────────────────────────────────────────────────────

@dataclasses.dataclass
class ModelSpec:
    """Specification for a single model output file."""
    name: str   # freeform label used in plots and directory names
    path: str   # path to .nc generation output file
    ckpt: str = ""  # optional checkpoint info, e.g. "800k", "1.4M"

    @classmethod
    def from_str(cls, s: str) -> "ModelSpec":
        """Parse 'name:path' or 'name:path:ckpt' string."""
        parts = s.split(":", maxsplit=2)
        if len(parts) < 2:
            raise ValueError(f"Model spec must be NAME:PATH[:CKPT], got: {s!r}")
        name, path = parts[0], parts[1]
        ckpt = parts[2] if len(parts) == 3 else ""
        return cls(name=name, path=path, ckpt=ckpt)

    @property
    def display_name(self) -> str:
        """Label for plot titles: 'name (step ckpt)' or just 'name'."""
        if self.ckpt:
            return f"{self.name} (step {self.ckpt})"
        return self.name


def parse_model_args(model_strs: list) -> list:
    """Parse a list of model spec strings.

    Args:
        model_strs: list of 'name:path[:ckpt]' strings

    Returns:
        list of ModelSpec objects
    """
    if not model_strs:
        raise ValueError("At least one --model argument is required.")
    return [ModelSpec.from_str(s) for s in model_strs]


def _spec_dir_name(spec) -> str:
    """Unique directory token: nc filename stem (no extension)."""
    return os.path.splitext(os.path.basename(spec.path))[0]


def make_output_dir(specs: list, base: str = RESULTS_BASE) -> str:
    """Build the output directory path from model specs.

    Single model:    base/ncfile
    Multiple models: base/ncfile_a_vs_ncfile_b_vs_...
    """
    dirname = "_vs_".join(_spec_dir_name(s) for s in specs)
    return os.path.join(base, dirname)


def assign_styles(specs: list) -> dict:
    """Assign colors and display labels to each model.

    Returns:
        dict mapping model name → {'color': str, 'label': str}
    """
    palette = _get_palette(len(specs))
    return {
        spec.name: {"color": palette[i], "label": spec.display_name}
        for i, spec in enumerate(specs)
    }


def comparison_title(specs: list) -> str:
    """Build a title suffix: 'model_a' or 'model_a vs model_b vs ...'."""
    return " vs ".join(s.display_name for s in specs)


# ─── Data loading ─────────────────────────────────────────────────────────────

def open_samples(path: str):
    """Open prediction and truth groups from a NetCDF4 generation output file.

    Returns:
        truth (xr.Dataset): ground truth with lat/lon coords
        pred  (xr.Dataset): predictions with ensemble dim, with lat/lon coords
        root  (xr.Dataset): root dataset (lat, lon, time)
    """
    root = xr.open_dataset(path)
    pred = xr.open_dataset(path, group="prediction")
    truth = xr.open_dataset(path, group="truth")

    pred = pred.merge(root)
    truth = truth.merge(root)

    truth = truth.set_coords(["lon", "lat"])
    pred = pred.set_coords(["lon", "lat"])
    return truth, pred, root


def add_wind_speed(ds: xr.Dataset) -> xr.Dataset:
    """Add wind_speed_10m = sqrt(u^2 + v^2) to dataset (returns copy)."""
    ws = np.sqrt(ds["eastward_wind_10m"] ** 2 + ds["northward_wind_10m"] ** 2)
    ws.attrs = {"long_name": "10m wind speed", "units": "m/s"}
    return ds.assign(wind_speed_10m=ws)


# ─── Math utilities ───────────────────────────────────────────────────────────

def pattern_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Spatial pattern correlation between two 2D arrays."""
    mx, my = x.mean(), y.mean()
    num = np.mean((x - mx) * (y - my))
    den = np.sqrt(np.mean((x - mx) ** 2) * np.mean((y - my) ** 2))
    return float(num / den) if den > 0 else 0.0


def haversine(lat1, lon1, lat2, lon2) -> float:
    """Haversine distance in meters between two lat/lon points."""
    R = 6371000.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
