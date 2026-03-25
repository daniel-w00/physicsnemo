"""Generate evaluation plots for CorrDiff generation outputs.

Requires metrics to be computed first (metrics.py). Then run:

    # Single model
    python analysis/generation/plots.py --model "reg:output/gen_taiwan/reg.nc:800k"

    # Two-model comparison
    python analysis/generation/plots.py \\
        --model "diffusion:output/gen_taiwan/diff.nc:1400k" \\
        --model "regression:output/gen_taiwan/reg.nc:800k"

Output saved to: analysis/generation/results/{name}/plots/
"""

import argparse
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.generation.utils import (
    ALL_VARS,
    OUTPUT_VARS,
    VAR_CMAP,
    VAR_LABELS,
    add_wind_speed,
    assign_styles,
    comparison_title,
    make_output_dir,
    open_samples,
    parse_model_args,
    pattern_correlation,
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _time_tick_labels(times: pd.DatetimeIndex, n_ticks: int = 10):
    """Evenly spaced tick positions and labels for sample-index x-axis."""
    n = len(times)
    idxs = np.linspace(0, n - 1, min(n_ticks, n), dtype=int)
    labels = [times[i].strftime("%m-%d\n%H:%M") for i in idxs]
    return idxs, labels


def _savefig(name: str, plots_dir: str):
    path = os.path.join(plots_dir, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def _load_metrics(spec, scores_dir: str, n_models: int) -> xr.Dataset:
    """Load cached metrics .nc for a given ModelSpec."""
    if n_models == 1:
        nc_path = os.path.join(scores_dir, "metrics.nc")
    else:
        nc_path = os.path.join(scores_dir, f"{spec.name}_metrics.nc")
    if not os.path.exists(nc_path):
        print(f"  ERROR: metrics file not found: {nc_path}")
        print("  Run metrics.py first with the same --model arguments.")
        sys.exit(1)
    return xr.open_dataset(nc_path)


def _ensure_2d_axs(axs, n_rows, n_cols):
    """Ensure axs is always shape (n_rows, n_cols) regardless of matplotlib's squeeze."""
    axs = np.array(axs)
    if axs.ndim == 0:
        axs = axs.reshape(1, 1)
    elif axs.ndim == 1:
        if n_rows == 1:
            axs = axs.reshape(1, -1)
        else:
            axs = axs.reshape(-1, 1)
    return axs


# ─── Plot 1 & 2: Metric time series ──────────────────────────────────────────

def plot_metric_timeseries(metric_name: str, specs, metrics_dict: dict, styles: dict, plots_dir: str):
    """Plot per-variable time series of a given metric for all models, sorted by time."""
    n = len(ALL_VARS)
    fig, axs = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
    axs = np.atleast_1d(axs)

    # Sort by time using first model as reference
    first_ds = next(iter(metrics_dict.values())).sortby("time")
    times = pd.to_datetime(first_ds["time"].values)
    x = np.arange(len(times))

    for ax, v in zip(axs, ALL_VARS):
        for spec in specs:
            ds = metrics_dict[spec.name].sortby("time")
            vals = ds[v].sel(metric=metric_name).values
            st = styles[spec.name]
            ax.plot(x, vals, color=st["color"], label=st["label"], linewidth=1.2)
        ax.set_ylabel(VAR_LABELS.get(v, v), fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    tick_pos, tick_labels = _time_tick_labels(times)
    axs[-1].set_xticks(tick_pos)
    axs[-1].set_xticklabels(tick_labels, fontsize=7)
    axs[-1].set_xlabel("Time (sorted)")

    title = f"{metric_name.upper()} per sample (spatially averaged) — {comparison_title(specs)}"
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    _savefig(f"{metric_name}_timeseries.png", plots_dir)


# ─── Plot 3 & 4: Spatial maps ─────────────────────────────────────────────────

def plot_spatial_map(map_type: str, specs, styles: dict, plots_dir: str):
    """Plot time-averaged spatial bias or MAE map for all models.

    map_type: 'bias' or 'error'
    """
    loop_vars = ALL_VARS
    n_vars = len(loop_vars)
    n_cols = len(specs)

    fig, axs = plt.subplots(n_vars, n_cols, figsize=(6 * n_cols, 4 * n_vars))
    axs = _ensure_2d_axs(axs, n_vars, n_cols)

    for col, spec in enumerate(specs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            truth_ds, pred_ds, root = open_samples(spec.path)
        truth_ds = add_wind_speed(truth_ds)
        pred_ds = add_wind_speed(pred_ds)
        lat = root["lat"].values
        lon = root["lon"].values

        for row, v in enumerate(loop_vars):
            pred_mean = pred_ds[v].mean("ensemble")
            ax = axs[row, col]

            if map_type == "bias":
                data_map = (pred_mean - truth_ds[v]).mean("time").values
                bound = max(abs(data_map.min()), abs(data_map.max()))
                im = ax.pcolormesh(lon, lat, data_map, cmap="RdBu_r",
                                   vmin=-bound, vmax=bound, shading="auto")
                title_prefix = "Bias"
            else:
                data_map = np.abs(pred_mean.values - truth_ds[v].values).mean(axis=0)
                im = ax.pcolormesh(lon, lat, data_map, cmap="YlOrRd", shading="auto")
                title_prefix = "MAE"

            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title(f"{styles[spec.name]['label']}\n{VAR_LABELS.get(v, v)}", fontsize=8)
            ax.set_xlabel("lon")
            ax.set_ylabel("lat")

    map_label = "Mean Bias (prediction − truth)" if map_type == "bias" else "Mean Absolute Error"
    fig.suptitle(f"{map_label}, time-averaged — {comparison_title(specs)}", fontsize=11)
    plt.tight_layout()
    _savefig(f"{map_type}_maps.png", plots_dir)


# ─── Plot 5: Spread vs Skill ──────────────────────────────────────────────────

def plot_spread_skill(specs, metrics_dict: dict, styles: dict, plots_dir: str):
    """Plot spread vs RMSE scatter for models with ensemble > 1.

    Skipped entirely if no model has ensemble > 1.
    """
    # Filter to ensemble models
    ens_specs = []
    for spec in specs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, pred_ds, _ = open_samples(spec.path)
        if pred_ds.sizes.get("ensemble", 1) > 1:
            ens_specs.append(spec)

    if not ens_specs:
        print("  Skipping spread/skill: no ensemble models.")
        return

    n_vars = len(ALL_VARS)
    ncols = 3
    nrows = int(np.ceil(n_vars / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axs_flat = np.array(axs).flat

    for v, ax in zip(ALL_VARS, axs_flat):
        for spec in ens_specs:
            ds = metrics_dict[spec.name]
            rmse = ds[v].sel(metric="rmse").values
            spread = ds[v].sel(metric="spread").values
            st = styles[spec.name]
            ax.scatter(rmse, spread, alpha=0.6, s=20, color=st["color"], label=st["label"])
        lim_max = max(
            max(metrics_dict[s.name][v].sel(metric="rmse").values.max() for s in ens_specs),
            max(metrics_dict[s.name][v].sel(metric="spread").values.max() for s in ens_specs),
        ) * 1.05
        ax.plot([0, lim_max], [0, lim_max], "k--", linewidth=1, label="ideal")
        ax.set_xlabel("RMSE")
        ax.set_ylabel("Ensemble Std. Dev. (adjusted)")
        ax.set_title(VAR_LABELS.get(v, v), fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    for ax in list(axs_flat)[n_vars:]:
        ax.set_visible(False)

    fig.suptitle(f"Spread vs. Skill — {comparison_title(ens_specs)}", fontsize=11)
    plt.tight_layout()
    _savefig("spread_skill.png", plots_dir)


# ─── Plot 6: Sample panels ────────────────────────────────────────────────────

def plot_sample_panels(specs, styles: dict, plots_dir: str, n_samples: int = 3,
                       events: dict = None):
    """Plot side-by-side spatial snapshots: truth + each model.

    Args:
        events: dict mapping date string (e.g. '2021-09-12') to event label
                (e.g. 'Typhoon Chanthu'). These timestamps are always included.
    """
    if events is None:
        events = {}

    vars_to_plot = OUTPUT_VARS
    n_vars = len(vars_to_plot)
    n_cols = 1 + len(specs)

    # Load all model data
    model_data = {}
    truth_data = None
    root_ref = None
    for spec in specs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            truth_ds, pred_ds, root = open_samples(spec.path)
        if truth_data is None:
            truth_data = truth_ds
            root_ref = root
        model_data[spec.name] = pred_ds

    lat = root_ref["lat"].values
    lon = root_ref["lon"].values
    # Sort everything by time so indices are chronological
    sort_order = np.argsort(root_ref["time"].values)
    times = pd.to_datetime(root_ref["time"].values[sort_order])
    n_time = len(times)

    # Build index list into the sorted array.
    # Forced event timestamps come first, fill the rest evenly.
    forced_sorted_idxs = []
    event_labels = {}  # sorted index -> event label
    for date_str, label in events.items():
        date = pd.Timestamp(date_str)
        matches = np.where(times.normalize() == date)[0]
        if len(matches) == 0:
            print(f"  WARNING: event date {date_str} not found in data, skipping.")
            continue
        idx = int(matches[0])
        forced_sorted_idxs.append(idx)
        event_labels[idx] = label

    remaining = n_samples - len(forced_sorted_idxs)
    if remaining > 0:
        fill_sorted_idxs = np.linspace(0, n_time - 1, remaining, dtype=int)
        fill_sorted_idxs = [i for i in fill_sorted_idxs if i not in forced_sorted_idxs][:remaining]
    else:
        fill_sorted_idxs = []

    # Final list of sorted indices, ordered chronologically
    all_sorted_idxs = sorted(set(forced_sorted_idxs + fill_sorted_idxs))

    for sorted_idx in all_sorted_idxs:
        # Map sorted index back to original file index for isel
        file_idx = int(sort_order[sorted_idx])

        fig, axs = plt.subplots(n_vars, n_cols, figsize=(5 * n_cols, 4 * n_vars))
        axs = _ensure_2d_axs(axs, n_vars, n_cols)

        # Column headers
        axs[0, 0].set_title("Truth", fontsize=10)
        for col, spec in enumerate(specs, start=1):
            axs[0, col].set_title(styles[spec.name]["label"], fontsize=10)

        for row, v in enumerate(vars_to_plot):
            tr = truth_data[v].isel(time=file_idx).values
            model_arrays = {
                spec.name: model_data[spec.name][v].isel(time=file_idx).mean("ensemble").values
                for spec in specs
            }
            all_arrays = [tr] + list(model_arrays.values())
            vmin = min(a.min() for a in all_arrays)
            vmax = max(a.max() for a in all_arrays)
            cmap = VAR_CMAP.get(v, "viridis")

            im = axs[row, 0].pcolormesh(lon, lat, tr, cmap=cmap,
                                         vmin=vmin, vmax=vmax, shading="auto")
            axs[row, 0].set_ylabel(VAR_LABELS.get(v, v), fontsize=8)

            for col, spec in enumerate(specs, start=1):
                arr = model_arrays[spec.name]
                im = axs[row, col].pcolormesh(lon, lat, arr, cmap=cmap,
                                               vmin=vmin, vmax=vmax, shading="auto")
                rmse = float(np.sqrt(np.mean((arr - tr) ** 2)))
                pc = pattern_correlation(arr, tr)
                axs[row, col].set_title(f"RMSE={rmse:.3f}  PC={pc:.2f}", fontsize=7)

                if col == len(specs):
                    plt.colorbar(im, ax=axs[row, col], fraction=0.046)

        time_str = times[sorted_idx].strftime("%Y-%m-%d %H:%M UTC")
        event_suffix = f" — {event_labels[sorted_idx]}" if sorted_idx in event_labels else ""
        fig.suptitle(f"Sample panel — {time_str}{event_suffix} — {comparison_title(specs)}", fontsize=10)
        plt.tight_layout()
        _savefig(f"sample_panel_{times[sorted_idx].strftime('%Y-%m-%d_%H%M')}.png", plots_dir)


# ─── Plot 7: Summary bar chart ────────────────────────────────────────────────

def plot_summary_bars(specs, metrics_dict: dict, styles: dict, plots_dir: str):
    """Grouped bar chart of time-mean metrics per variable, one chart per metric."""
    n_models = len(specs)
    width = 0.8 / n_models
    x = np.arange(len(ALL_VARS))
    var_labels = [VAR_LABELS.get(v, v) for v in ALL_VARS]

    for metric in ["rmse", "mae", "crps"]:
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, spec in enumerate(specs):
            ds = metrics_dict[spec.name]
            vals = [float(ds[v].sel(metric=metric).mean("time").values) for v in ALL_VARS]
            offset = (i - (n_models - 1) / 2) * width
            st = styles[spec.name]
            ax.bar(x + offset, vals, width, label=st["label"], color=st["color"])

        ax.set_xticks(x)
        ax.set_xticklabels(var_labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Time-mean {metric.upper()} — {comparison_title(specs)}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        _savefig(f"summary_bar_{metric}.png", plots_dir)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation plots for CorrDiff generation outputs."
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
    parser.add_argument(
        "--n-samples", type=int, default=3,
        help="Number of sample panel timesteps (default: 3)",
    )
    parser.add_argument(
        "--event", action="append", default=[],
        metavar="DATE:LABEL",
        help="Force a timestep into sample panels with an event label, e.g. '2021-09-12:Typhoon Chanthu'. Repeatable.",
    )
    args = parser.parse_args()

    specs = parse_model_args(args.model)
    out_dir = make_output_dir(specs, base=args.outdir) if args.outdir else make_output_dir(specs)
    scores_dir = os.path.join(out_dir, "scores")
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    styles = assign_styles(specs)

    # Load cached metrics
    print("Loading metrics...")
    metrics_dict = {spec.name: _load_metrics(spec, scores_dir, len(specs)) for spec in specs}

    print("Plot 1: RMSE time series")
    plot_metric_timeseries("rmse", specs, metrics_dict, styles, plots_dir)

    print("Plot 2: CRPS time series")
    plot_metric_timeseries("crps", specs, metrics_dict, styles, plots_dir)

    print("Plot 3: Bias maps")
    plot_spatial_map("bias", specs, styles, plots_dir)

    print("Plot 4: Error maps")
    plot_spatial_map("error", specs, styles, plots_dir)

    print("Plot 5: Spread vs Skill")
    plot_spread_skill(specs, metrics_dict, styles, plots_dir)

    # Parse --event DATE:LABEL args
    events = {}
    for e in args.event:
        parts = e.split(":", maxsplit=1)
        if len(parts) != 2:
            print(f"  WARNING: --event must be DATE:LABEL, got {e!r}, skipping.")
            continue
        events[parts[0].strip()] = parts[1].strip()

    print("Plot 6: Sample panels")
    plot_sample_panels(specs, styles, plots_dir, n_samples=args.n_samples, events=events)

    print("Plot 7: Summary bar charts")
    plot_summary_bars(specs, metrics_dict, styles, plots_dir)

    print(f"\nAll plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
