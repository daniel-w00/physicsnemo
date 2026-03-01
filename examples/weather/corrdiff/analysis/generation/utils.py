"""Shared utilities for generation output analysis."""

import numpy as np
import xarray as xr


# Paths to the two output files (relative to repo root)
DIFFUSION_FILE = "output/gen_taiwan/all4ens-2021-02-02-and03-1week-3h.nc"
REGRESSION_FILE = "output/gen_taiwan/regonly-2021-02-02-and03-1week-3h.nc"

# Variables in truth/prediction groups
OUTPUT_VARS = [
    "maximum_radar_reflectivity",
    "temperature_2m",
    "eastward_wind_10m",
    "northward_wind_10m",
]

VAR_LABELS = {
    "maximum_radar_reflectivity": "Reflectivity [dBZ]",
    "temperature_2m": "T2m [K]",
    "eastward_wind_10m": "U-wind 10m [m/s]",
    "northward_wind_10m": "V-wind 10m [m/s]",
    "wind_speed_10m": "Wind Speed 10m [m/s]",
}


def open_samples(path: str):
    """Open prediction and truth groups from a NetCDF4 file.

    Returns:
        truth (xr.Dataset): ground truth with lat/lon coords
        pred  (xr.Dataset): predictions with ensemble dim, with lat/lon coords
        root  (xr.Dataset): root dataset (lat, lon, time)
    """
    root = xr.open_dataset(path)
    pred = xr.open_dataset(path, group="prediction")
    truth = xr.open_dataset(path, group="truth")

    # merge root coords (lat, lon, time) into sub-groups
    pred = pred.merge(root)
    truth = truth.merge(root)

    truth = truth.set_coords(["lon", "lat"])
    pred = pred.set_coords(["lon", "lat"])
    return truth, pred, root


def add_wind_speed(ds: xr.Dataset) -> xr.Dataset:
    """Add wind_speed_10m = sqrt(u^2 + v^2) to dataset in-place (returns copy)."""
    ws = np.sqrt(ds["eastward_wind_10m"] ** 2 + ds["northward_wind_10m"] ** 2)
    ws.attrs = {"long_name": "10m wind speed", "units": "m/s"}
    return ds.assign(wind_speed_10m=ws)


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
