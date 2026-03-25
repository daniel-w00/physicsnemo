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
    add_wind_speed,
    make_output_dir,
    open_samples,
    parse_model_args,
    pattern_correlation,
)


def crps_ensemble_numpy(truth: np.ndarray, ensemble: np.ndarray) -> np.ndarray:
    """CRPS for ensemble forecasts, spatially averaged over last two dims.

    Args:
        truth:    array of shape (..., y, x)
        ensemble: array of shape (n_members, ..., y, x)

    Returns:
        CRPS spatially averaged, shape (...)
    """
    n = ensemble.shape[0]
    mae = np.mean(np.abs(ensemble.mean(axis=0) - truth), axis=(-2, -1))
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

    truth_ds = add_wind_speed(truth_ds)
    pred_ds = add_wind_speed(pred_ds)

    n_time = truth_ds.sizes["time"]
    times = root["time"].values

    metrics_list = []

    for v in ALL_VARS:
        truth_arr = truth_ds[v].values   # (time, y, x)
        pred_arr = pred_ds[v].values     # (ensemble, time, y, x)
        pred_mean = pred_arr.mean(axis=0)
        n_ens = pred_arr.shape[0]

        rmse = np.sqrt(np.mean((pred_mean - truth_arr) ** 2, axis=(-2, -1)))
        mae = np.mean(np.abs(pred_mean - truth_arr), axis=(-2, -1))
        bias = np.mean(pred_mean - truth_arr, axis=(-2, -1))
        # Adjusted spread: multiply by √(1 + 1/n) so that spread == RMSE
        # at perfect calibration (accounts for ensemble mean being excluded)
        spread = pred_arr.std(axis=0).mean(axis=(-2, -1)) * np.sqrt(1 + 1 / n_ens)
        spread_skill = spread / (rmse + 1e-10)

        pc = np.array([
            pattern_correlation(pred_mean[t], truth_arr[t]) for t in range(n_time)
        ])

        if n_ens > 1:
            crps = crps_ensemble_numpy(truth_arr, pred_arr)
        else:
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

    ds = xr.Dataset({v: m for v, m in zip(ALL_VARS, metrics_list)})
    ds.attrs["source_file"] = path
    ds.attrs["model"] = label
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
