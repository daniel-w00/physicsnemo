"""Generate evaluation plots for CorrDiff generation outputs.

Requires metrics to have been computed first:
    python analysis/generation/metrics.py

Then run from repo root:
    python analysis/generation/plots.py

Outputs saved to: analysis/generation/plots/
"""

import os
import sys
import warnings

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import xarray as xr

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
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")

ALL_VARS = OUTPUT_VARS + ["wind_speed_10m"]

# Color / style for the two models
MODEL_STYLE = {
    "diffusion": {"color": "#e07b39", "label": "Diffusion (4-ens)"},
    "regression": {"color": "#3b7dd8", "label": "Regression (det.)"},
}

# Colormaps per variable for spatial maps
VAR_CMAP = {
    "maximum_radar_reflectivity": "magma",
    "temperature_2m": "RdYlBu_r",
    "eastward_wind_10m": "RdBu_r",
    "northward_wind_10m": "RdBu_r",
    "wind_speed_10m": "viridis",
}


def load_metrics():
    diff = xr.open_dataset(os.path.join(SCORES_DIR, "diffusion_metrics.nc"))
    reg = xr.open_dataset(os.path.join(SCORES_DIR, "regression_metrics.nc"))
    return diff, reg


def savefig(name: str):
    path = os.path.join(PLOTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── Plot 1: RMSE time series ────────────────────────────────────────────────

def _time_tick_labels(times: pd.DatetimeIndex, n_ticks: int = 10):
    """Return evenly-spaced tick positions and labels for sample-index x-axis."""
    n = len(times)
    idxs = np.linspace(0, n - 1, min(n_ticks, n), dtype=int)
    labels = [times[i].strftime("%m-%d\n%H:%M") for i in idxs]
    return idxs, labels


def plot_rmse_timeseries(diff, reg):
    vars_to_plot = [v for v in ALL_VARS if v != "wind_speed_10m"] + ["wind_speed_10m"]
    n = len(vars_to_plot)
    fig, axs = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
    # Use sample index as x so unsorted timestamps don't distort the line
    times = pd.to_datetime(diff["time"].values)
    x = np.arange(len(times))

    for ax, v in zip(axs, vars_to_plot):
        for model, ds in [("diffusion", diff), ("regression", reg)]:
            vals = ds[v].sel(metric="rmse").values
            st = MODEL_STYLE[model]
            ax.plot(x, vals, color=st["color"], label=st["label"], linewidth=1.2)
        ax.set_ylabel(VAR_LABELS.get(v, v), fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    tick_pos, tick_labels = _time_tick_labels(times)
    axs[-1].set_xticks(tick_pos)
    axs[-1].set_xticklabels(tick_labels, fontsize=7)
    axs[-1].set_xlabel("Sample (timestamp in file order)")
    fig.suptitle("RMSE per sample (spatially averaged)", fontsize=12)
    plt.tight_layout()
    savefig("rmse_timeseries.png")


# ─── Plot 2: CRPS time series ────────────────────────────────────────────────

def plot_crps_timeseries(diff, reg):
    vars_to_plot = ALL_VARS
    n = len(vars_to_plot)
    fig, axs = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
    times = pd.to_datetime(diff["time"].values)
    x = np.arange(len(times))

    for ax, v in zip(axs, vars_to_plot):
        for model, ds in [("diffusion", diff), ("regression", reg)]:
            vals = ds[v].sel(metric="crps").values
            st = MODEL_STYLE[model]
            ax.plot(x, vals, color=st["color"], label=st["label"], linewidth=1.2)
        ax.set_ylabel(VAR_LABELS.get(v, v), fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    tick_pos, tick_labels = _time_tick_labels(times)
    axs[-1].set_xticks(tick_pos)
    axs[-1].set_xticklabels(tick_labels, fontsize=7)
    axs[-1].set_xlabel("Sample (timestamp in file order)")
    fig.suptitle("CRPS per sample (spatially averaged)", fontsize=12)
    plt.tight_layout()
    savefig("crps_timeseries.png")


# ─── Plot 3: Bias maps ───────────────────────────────────────────────────────

def plot_bias_maps():
    datasets = {
        "diffusion": open_samples(DIFFUSION_FILE),
        "regression": open_samples(REGRESSION_FILE),
    }
    vars_to_plot = OUTPUT_VARS  # wind_speed added below
    n_vars = len(vars_to_plot) + 1  # +1 for wind speed
    n_models = 2
    fig, axs = plt.subplots(n_vars, n_models, figsize=(10, 4 * n_vars))

    for col, (model, (truth_ds, pred_ds, root)) in enumerate(datasets.items()):
        truth_ds = add_wind_speed(truth_ds)
        pred_ds = add_wind_speed(pred_ds)
        loop_vars = OUTPUT_VARS + ["wind_speed_10m"]

        for row, v in enumerate(loop_vars):
            pred_mean = pred_ds[v].mean("ensemble")  # (time, y, x)
            bias_map = (pred_mean - truth_ds[v]).mean("time").values  # (y, x)
            lat = root["lat"].values
            lon = root["lon"].values

            ax = axs[row, col]
            bound = max(abs(bias_map.min()), abs(bias_map.max()))
            im = ax.pcolormesh(lon, lat, bias_map, cmap="RdBu_r",
                               vmin=-bound, vmax=bound, shading="auto")
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title(f"{MODEL_STYLE[model]['label']}\n{VAR_LABELS.get(v, v)}", fontsize=8)
            ax.set_xlabel("lon")
            ax.set_ylabel("lat")

    fig.suptitle("Mean Bias (prediction − truth), averaged over time", fontsize=12)
    plt.tight_layout()
    savefig("bias_maps.png")


# ─── Plot 4: Error maps (mean absolute error spatially) ─────────────────────

def plot_error_maps():
    datasets = {
        "diffusion": open_samples(DIFFUSION_FILE),
        "regression": open_samples(REGRESSION_FILE),
    }
    loop_vars = OUTPUT_VARS + ["wind_speed_10m"]
    n_vars = len(loop_vars)
    n_models = 2
    fig, axs = plt.subplots(n_vars, n_models, figsize=(10, 4 * n_vars))

    for col, (model, (truth_ds, pred_ds, root)) in enumerate(datasets.items()):
        truth_ds = add_wind_speed(truth_ds)
        pred_ds = add_wind_speed(pred_ds)
        lat = root["lat"].values
        lon = root["lon"].values

        for row, v in enumerate(loop_vars):
            pred_mean = pred_ds[v].mean("ensemble")
            mae_map = np.abs(pred_mean.values - truth_ds[v].values).mean(axis=0)

            ax = axs[row, col]
            im = ax.pcolormesh(lon, lat, mae_map, cmap="YlOrRd", shading="auto")
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title(f"{MODEL_STYLE[model]['label']}\n{VAR_LABELS.get(v, v)}", fontsize=8)
            ax.set_xlabel("lon")
            ax.set_ylabel("lat")

    fig.suptitle("Mean Absolute Error (time-averaged spatial map)", fontsize=12)
    plt.tight_layout()
    savefig("error_maps.png")


# ─── Plot 5: Spread vs Skill (diffusion only) ────────────────────────────────

def plot_spread_skill(diff):
    n_vars = len(ALL_VARS)
    ncols = 3
    nrows = int(np.ceil(n_vars / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axs_flat = axs.flat

    for v, ax in zip(ALL_VARS, axs_flat):
        rmse = diff[v].sel(metric="rmse").values
        spread = diff[v].sel(metric="spread").values
        ax.scatter(spread, rmse, alpha=0.6, s=20, color=MODEL_STYLE["diffusion"]["color"])
        lim = max(rmse.max(), spread.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="ideal (ratio=1)")
        ax.set_xlabel("Ensemble Spread")
        ax.set_ylabel("RMSE")
        ax.set_title(VAR_LABELS.get(v, v), fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    # hide unused axes
    for ax in list(axs_flat)[n_vars:]:
        ax.set_visible(False)

    fig.suptitle("Spread vs. Skill — Diffusion model (one point per timestep)", fontsize=12)
    plt.tight_layout()
    savefig("spread_skill.png")


# ─── Plot 6: Sample panels ───────────────────────────────────────────────────

def plot_sample_panels(n_samples: int = 3):
    truth_diff, pred_diff, root_diff = open_samples(DIFFUSION_FILE)
    truth_reg, pred_reg, _ = open_samples(REGRESSION_FILE)

    lat = root_diff["lat"].values
    lon = root_diff["lon"].values
    times = pd.to_datetime(root_diff["time"].values)
    n_time = len(times)

    # pick evenly spaced timestep indices
    idxs = np.linspace(0, n_time - 1, n_samples, dtype=int)

    vars_to_plot = OUTPUT_VARS
    n_vars = len(vars_to_plot)

    for t_idx in idxs:
        fig, axs = plt.subplots(n_vars, 3, figsize=(13, 4 * n_vars))
        axs[0, 0].set_title("Truth", fontsize=10)
        axs[0, 1].set_title("Regression (pred)", fontsize=10)
        axs[0, 2].set_title("Diffusion (ens-mean)", fontsize=10)

        for row, v in enumerate(vars_to_plot):
            tr = truth_diff[v].isel(time=t_idx).values
            rg = pred_reg[v].isel(time=t_idx).mean("ensemble").values
            df = pred_diff[v].isel(time=t_idx).mean("ensemble").values

            vmin = min(tr.min(), rg.min(), df.min())
            vmax = max(tr.max(), rg.max(), df.max())
            cmap = VAR_CMAP.get(v, "viridis")

            for col, (data, label) in enumerate([(tr, "truth"), (rg, "reg"), (df, "diff")]):
                ax = axs[row, col]
                im = ax.pcolormesh(lon, lat, data, cmap=cmap,
                                   vmin=vmin, vmax=vmax, shading="auto")
                if col == 2:
                    plt.colorbar(im, ax=ax, fraction=0.046)

            # annotate RMSE
            rmse_reg = float(np.sqrt(np.mean((rg - tr) ** 2)))
            rmse_diff = float(np.sqrt(np.mean((df - tr) ** 2)))
            pc_reg = pattern_correlation(rg, tr)
            pc_diff = pattern_correlation(df, tr)

            axs[row, 0].set_ylabel(VAR_LABELS.get(v, v), fontsize=8)
            axs[row, 1].set_title(f"Regression  RMSE={rmse_reg:.3f}  PC={pc_reg:.2f}", fontsize=7)
            axs[row, 2].set_title(f"Diffusion ens-mean  RMSE={rmse_diff:.3f}  PC={pc_diff:.2f}", fontsize=7)

        time_str = times[t_idx].strftime("%Y-%m-%d %H:%M UTC")
        fig.suptitle(f"Sample panel — {time_str}", fontsize=11)
        plt.tight_layout()
        fname = f"sample_panel_t{t_idx:03d}.png"
        savefig(fname)


# ─── Plot 8: Summary bar chart ───────────────────────────────────────────────

def plot_summary_bar():
    summary = pd.read_csv(os.path.join(SCORES_DIR, "summary.csv"))

    for metric in ["rmse", "crps", "mae"]:
        subset = summary[summary["metric"] == metric]
        pivot = subset.pivot_table(index="variable", columns="model", values="value")

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(pivot))
        width = 0.35

        bars_diff = pivot.get("diffusion", pd.Series(dtype=float))
        bars_reg = pivot.get("regression", pd.Series(dtype=float))

        ax.bar(x - width / 2, bars_diff.values, width,
               label="Diffusion", color=MODEL_STYLE["diffusion"]["color"])
        ax.bar(x + width / 2, bars_reg.values, width,
               label="Regression", color=MODEL_STYLE["regression"]["color"])

        ax.set_xticks(x)
        ax.set_xticklabels(
            [VAR_LABELS.get(v, v) for v in pivot.index], rotation=20, ha="right", fontsize=9
        )
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Time-mean {metric.upper()} — Diffusion vs Regression")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        savefig(f"summary_bar_{metric}.png")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Load pre-computed metrics
    if not os.path.exists(os.path.join(SCORES_DIR, "diffusion_metrics.nc")):
        print("Metrics not found. Run metrics.py first.")
        sys.exit(1)

    print("Loading metrics...")
    diff, reg = load_metrics()

    print("Plot 1: RMSE time series")
    plot_rmse_timeseries(diff, reg)

    print("Plot 2: CRPS time series")
    plot_crps_timeseries(diff, reg)

    print("Plot 3: Bias maps")
    plot_bias_maps()

    print("Plot 4: Error maps")
    plot_error_maps()

    print("Plot 5: Spread vs Skill")
    plot_spread_skill(diff)

    print("Plot 6: Sample panels")
    plot_sample_panels(n_samples=3)

    print("Plot 7: Summary bar charts")
    plot_summary_bar()

    print(f"\nAll plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
