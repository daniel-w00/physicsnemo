"""Compute evaluation metrics for CorrDiff generation outputs.

Computes per-variable metrics (RMSE, MAE, Bias, CRPS, Spread, Spread/Skill,
Pattern Correlation) for one or two model output files and saves results to:
    analysis/generation/results/{name}/scores/metrics.nc        (single model)
    analysis/generation/results/{a}_vs_{b}/scores/{name}_metrics.nc  (comparison)

Run from the repo root:
    # Single model
    python analysis/generation/metrics.py --model "reg:output/gen_taiwan/reg.nc:800k"

    # Two-model comparison
    python analysis/generation/metrics.py \\
        --model "diffusion:output/gen_taiwan/diff.nc:1400k" \\
        --model "regression:output/gen_taiwan/reg.nc:800k"
"""

import argparse
import os
import sys
import warnings

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.generation.utils import (
    ALL_VARS,
    OUTPUT_VARS,
    make_output_dir,
    open_samples,
    parse_model_args,
    pattern_correlation,
)

_CHUNK = 50   # timesteps per chunk — keeps peak RAM < ~300 MB per variable


def _load_var(ds: xr.Dataset, var: str, t_slice) -> np.ndarray:
    """Load a time slice of var; computes wind_speed_10m from u/v on the fly."""
    if var == "wind_speed_10m":
        u = ds["eastward_wind_10m"].isel(time=t_slice).values
        v = ds["northward_wind_10m"].isel(time=t_slice).values
        return np.sqrt(u ** 2 + v ** 2)
    return ds[var].isel(time=t_slice).values


def crps_ensemble_numpy(truth: np.ndarray, ensemble: np.ndarray) -> np.ndarray:
    """CRPS for ensemble forecasts, spatially averaged over last two dims.

    Args:
        truth:    array of shape (..., y, x)
        ensemble: array of shape (n_members, ..., y, x)

    Returns:
        CRPS spatially averaged, shape (...)
    """
    n = ensemble.shape[0]
    # E[|X - y|]: average over members first, then space — NOT |mean - y|
    mae_term = np.mean(np.abs(ensemble - truth[np.newaxis, ...]), axis=(0, -2, -1))
    # E[|X - X'|]: average over unique pairs and space
    spread_sum = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            spread_sum += np.mean(np.abs(ensemble[i] - ensemble[j]), axis=(-2, -1))
    spread_term = spread_sum / (n * (n - 1) / 2)
    return mae_term - 0.5 * spread_term


def load_metrics(spec, scores_dir: str, n_models: int) -> xr.Dataset:
    """Load cached metrics NetCDF for a given ModelSpec.

    Args:
        spec:       ModelSpec instance
        scores_dir: directory containing the metrics .nc files
        n_models:   total number of models (determines filename)

    Raises:
        FileNotFoundError: if the metrics file has not been computed yet
    """
    if n_models == 1:
        nc_path = os.path.join(scores_dir, "metrics.nc")
    else:
        nc_path = os.path.join(scores_dir, f"{spec.name}_metrics.nc")
    if not os.path.exists(nc_path):
        raise FileNotFoundError(
            f"Metrics file not found: {nc_path}\n"
            "Run metrics.py first with the same --model arguments."
        )
    return xr.open_dataset(nc_path)


def compute_metrics_for_file(path: str, label: str) -> xr.Dataset:
    """Compute all metrics for a single output file.

    Returns an xr.Dataset with dims (metric, time) for each variable.
    Processes data in time chunks to avoid loading large arrays into memory.
    """
    truth_ds, pred_ds, root = open_samples(path)

    n_time = truth_ds.sizes["time"]
    times = root["time"].values
    n_ens = pred_ds.sizes["ensemble"]

    lat_arr = root["lat"].values
    lon_arr = root["lon"].values
    if lat_arr.ndim == 1 and lon_arr.ndim == 1:
        lon_2d, lat_2d = np.meshgrid(lon_arr, lat_arr)
    else:
        lat_2d, lon_2d = lat_arr, lon_arr

    # Determine spatial shape without loading full array
    n_y, n_x = truth_ds[OUTPUT_VARS[0]].isel(time=0).shape

    metrics_list = []
    bias_maps = {}
    mae_maps = {}

    for v in ALL_VARS:
        rmse         = np.zeros(n_time)
        mae          = np.zeros(n_time)
        bias         = np.zeros(n_time)
        crps         = np.zeros(n_time)
        spread       = np.zeros(n_time)
        spread_skill = np.zeros(n_time)
        pc           = np.zeros(n_time)
        bias_map_sum = np.zeros((n_y, n_x))
        mae_map_sum  = np.zeros((n_y, n_x))

        for t0 in range(0, n_time, _CHUNK):
            t1 = min(t0 + _CHUNK, n_time)
            sl = slice(t0, t1)

            truth_c = _load_var(truth_ds, v, sl)   # (chunk, y, x)
            pred_c  = _load_var(pred_ds,  v, sl)   # (ens, chunk, y, x)
            pmean   = pred_c.mean(axis=0)           # (chunk, y, x)
            diff    = pmean - truth_c

            rmse[t0:t1]         = np.sqrt(np.mean(diff ** 2,     axis=(-2, -1)))
            mae[t0:t1]          = np.mean(np.abs(diff),           axis=(-2, -1))
            bias[t0:t1]         = np.mean(diff,                   axis=(-2, -1))
            spread[t0:t1]       = pred_c.std(axis=0).mean(axis=(-2, -1))
            spread_skill[t0:t1] = spread[t0:t1] / (rmse[t0:t1] + 1e-10)
            crps[t0:t1]         = crps_ensemble_numpy(truth_c, pred_c) if n_ens > 1 else mae[t0:t1]

            for i in range(t1 - t0):
                pc[t0 + i] = pattern_correlation(pmean[i], truth_c[i])

            bias_map_sum += diff.sum(axis=0)
            mae_map_sum  += np.abs(diff).sum(axis=0)

        da = xr.DataArray(
            data=np.stack([rmse, mae, bias, crps, spread, spread_skill, pc], axis=0),
            dims=["metric", "time"],
            coords={
                "metric": ["rmse", "mae", "bias", "crps", "spread", "spread_skill", "pattern_corr"],
                "time": times,
            },
            name=v,
        )
        metrics_list.append(da)

        bias_maps[v] = bias_map_sum / n_time
        mae_maps[v]  = mae_map_sum  / n_time

    ds = xr.Dataset({v: m for v, m in zip(ALL_VARS, metrics_list)})

    for v in ALL_VARS:
        ds[f"{v}_bias_map"] = xr.DataArray(bias_maps[v], dims=["y", "x"])
        ds[f"{v}_mae_map"]  = xr.DataArray(mae_maps[v],  dims=["y", "x"])
    ds["lat"] = xr.DataArray(lat_2d, dims=["y", "x"])
    ds["lon"] = xr.DataArray(lon_2d, dims=["y", "x"])

    ds.attrs["source_file"] = path
    ds.attrs["model"]       = label
    ds.attrs["n_ensemble"]  = n_ens
    return ds


def main():
    parser = argparse.ArgumentParser(
        description="Compute evaluation metrics for CorrDiff generation outputs."
    )
    parser.add_argument(
        "--model", action="append", required=True,
        metavar="NAME:PATH[:CKPT]",
        help="Model spec: name:path[:checkpoint_step]. Repeat for two-model comparison.",
    )
    parser.add_argument(
        "--outdir", default=None,
        help="Override base output directory (default: analysis/generation/results/)",
    )
    args = parser.parse_args()

    specs = parse_model_args(args.model)
    out_dir = make_output_dir(specs, base=args.outdir) if args.outdir else make_output_dir(specs)
    scores_dir = os.path.join(out_dir, "scores")
    os.makedirs(scores_dir, exist_ok=True)

    for spec in specs:
        print(f"Computing metrics: {spec.display_name}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = compute_metrics_for_file(spec.path, spec.name)

        ds.attrs["checkpoint"] = spec.ckpt

        if len(specs) == 1:
            nc_path = os.path.join(scores_dir, "metrics.nc")
        else:
            nc_path = os.path.join(scores_dir, f"{spec.name}_metrics.nc")

        ds.to_netcdf(nc_path)
        print(f"  Saved: {nc_path}")


if __name__ == "__main__":
    main()
