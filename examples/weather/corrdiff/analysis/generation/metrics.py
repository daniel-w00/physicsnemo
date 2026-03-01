"""Compute evaluation metrics for CorrDiff generation outputs.

Computes per-variable metrics for both the diffusion (4-ensemble) and
regression (1-ensemble) output files and saves results to:
  analysis/generation/scores/diffusion_metrics.nc
  analysis/generation/scores/regression_metrics.nc
  analysis/generation/scores/summary.csv

Run from the repo root:
    python analysis/generation/metrics.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import xarray as xr

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.generation.utils import (
    DIFFUSION_FILE,
    OUTPUT_VARS,
    REGRESSION_FILE,
    VAR_LABELS,
    add_wind_speed,
    open_samples,
    pattern_correlation,
)

SCORES_DIR = os.path.join(os.path.dirname(__file__), "scores")


def crps_ensemble_numpy(truth: np.ndarray, ensemble: np.ndarray) -> np.ndarray:
    """CRPS for ensemble forecasts, computed over last two spatial dims.

    Args:
        truth: array of shape (..., y, x)
        ensemble: array of shape (n_members, ..., y, x)

    Returns:
        CRPS spatially averaged, shape (...)
    """
    n = ensemble.shape[0]
    # Mean absolute error of ensemble mean vs truth
    mae = np.mean(np.abs(ensemble.mean(axis=0) - truth), axis=(-2, -1))
    # Ensemble spread term: E[|X - X'|] / 2
    spread = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            spread += np.mean(np.abs(ensemble[i] - ensemble[j]), axis=(-2, -1))
    spread = spread / (n * (n - 1) / 2)
    return mae - 0.5 * spread


def compute_metrics_for_file(path: str, label: str) -> xr.Dataset:
    """Compute all metrics for a single output file.

    Returns an xr.Dataset with dims (metric, time) for each variable.
    """
    truth_ds, pred_ds, root = open_samples(path)

    # Add wind speed
    truth_ds = add_wind_speed(truth_ds)
    pred_ds = add_wind_speed(pred_ds)

    all_vars = OUTPUT_VARS + ["wind_speed_10m"]
    n_time = truth_ds.sizes["time"]
    times = root["time"].values

    metrics_list = []

    for v in all_vars:
        truth_arr = truth_ds[v].values  # (time, y, x)
        pred_arr = pred_ds[v].values    # (ensemble, time, y, x)
        pred_mean = pred_arr.mean(axis=0)  # (time, y, x)
        n_ens = pred_arr.shape[0]

        rmse = np.sqrt(np.mean((pred_mean - truth_arr) ** 2, axis=(-2, -1)))  # (time,)
        mae = np.mean(np.abs(pred_mean - truth_arr), axis=(-2, -1))
        bias = np.mean(pred_mean - truth_arr, axis=(-2, -1))
        spread = pred_arr.std(axis=0).mean(axis=(-2, -1))  # (time,)
        spread_skill = spread / (rmse + 1e-10)

        # Pattern correlation per timestep (ensemble mean vs truth)
        pc = np.array([
            pattern_correlation(pred_mean[t], truth_arr[t]) for t in range(n_time)
        ])

        # CRPS
        if n_ens > 1:
            crps = crps_ensemble_numpy(truth_arr, pred_arr)
        else:
            # For single-member: CRPS = MAE
            crps = mae.copy()

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

    ds = xr.Dataset({v: m for v, m in zip(all_vars, metrics_list)})
    ds.attrs["source_file"] = path
    ds.attrs["model"] = label
    return ds


def main():
    os.makedirs(SCORES_DIR, exist_ok=True)

    print("Computing metrics for diffusion model...")
    diff_metrics = compute_metrics_for_file(DIFFUSION_FILE, "diffusion")
    diff_out = os.path.join(SCORES_DIR, "diffusion_metrics.nc")
    diff_metrics.to_netcdf(diff_out)
    print(f"  Saved: {diff_out}")

    print("Computing metrics for regression model...")
    reg_metrics = compute_metrics_for_file(REGRESSION_FILE, "regression")
    reg_out = os.path.join(SCORES_DIR, "regression_metrics.nc")
    reg_metrics.to_netcdf(reg_out)
    print(f"  Saved: {reg_out}")

    # Build summary CSV: time-mean scalars for each model x variable x metric
    rows = []
    all_vars = OUTPUT_VARS + ["wind_speed_10m"]
    for model, ds in [("diffusion", diff_metrics), ("regression", reg_metrics)]:
        for v in all_vars:
            for metric in ds[v].metric.values:
                val = float(ds[v].sel(metric=metric).mean("time").values)
                rows.append({
                    "model": model,
                    "variable": v,
                    "metric": metric,
                    "value": val,
                })

    summary = pd.DataFrame(rows)
    summary_out = os.path.join(SCORES_DIR, "summary.csv")
    summary.to_csv(summary_out, index=False)
    print(f"  Saved: {summary_out}")

    # Print a readable summary table
    print("\n=== Time-mean scores (spatial average) ===")
    pivot = summary[summary["metric"].isin(["rmse", "mae", "crps"])].pivot_table(
        index=["variable", "metric"], columns="model", values="value"
    )
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(pivot.to_string())


if __name__ == "__main__":
    main()
