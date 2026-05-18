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
# add_wind_speed still used by plot_rank_histogram and plot_sample_panels
from analysis.generation.metrics import load_metrics

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
    if metric_name == "crps":
        specs = [s for s in specs if metrics_dict[s.name].attrs.get("n_ensemble", 1) > 1]
        if not specs:
            print("  Skipping CRPS time series: no ensemble models.")
            return

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

def plot_spatial_map(map_type: str, specs, metrics_dict: dict, styles: dict, plots_dir: str):
    """Plot time-averaged spatial bias or MAE map for all models.

    Uses pre-computed maps from metrics_dict (no raw-file re-reads).
    map_type: 'bias' or 'error'
    """
    n_vars = len(ALL_VARS)
    n_cols = len(specs)

    fig, axs = plt.subplots(n_vars, n_cols, figsize=(6 * n_cols, 4 * n_vars))
    axs = _ensure_2d_axs(axs, n_vars, n_cols)

    for col, spec in enumerate(specs):
        ds = metrics_dict[spec.name]
        lat = ds["lat"].values
        lon = ds["lon"].values

        for row, v in enumerate(ALL_VARS):
            ax = axs[row, col]

            if map_type == "bias":
                data_map = ds[f"{v}_bias_map"].values
                bound = max(abs(data_map.min()), abs(data_map.max()))
                im = ax.pcolormesh(lon, lat, data_map, cmap="RdBu_r",
                                   vmin=-bound, vmax=bound, shading="auto")
            else:
                data_map = ds[f"{v}_mae_map"].values
                im = ax.pcolormesh(lon, lat, data_map, cmap="YlOrRd", shading="auto")

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
    # Filter to ensemble models using cached attribute — no raw-file re-read
    ens_specs = [
        spec for spec in specs
        if metrics_dict[spec.name].attrs.get("n_ensemble", 1) > 1
    ]

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
    x = np.arange(len(ALL_VARS))
    var_labels = [VAR_LABELS.get(v, v) for v in ALL_VARS]

    for metric in ["rmse", "mae", "crps"]:
        plot_specs = specs if metric != "crps" else [
            s for s in specs if metrics_dict[s.name].attrs.get("n_ensemble", 1) > 1
        ]
        if not plot_specs:
            print(f"  Skipping {metric.upper()} bar chart: no ensemble models.")
            continue
        n_plot = len(plot_specs)
        width = 0.8 / n_plot
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, spec in enumerate(plot_specs):
            ds = metrics_dict[spec.name]
            vals = [float(ds[v].sel(metric=metric).mean("time").values) for v in ALL_VARS]
            offset = (i - (n_plot - 1) / 2) * width
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


# ─── Plot 8: Rank (Talagrand) histogram ──────────────────────────────────────

def plot_rank_histogram(specs, styles: dict, plots_dir: str):
    """Talagrand (rank) histogram for ensemble calibration.

    For each grid point and timestep the rank of the truth value among the
    ensemble members is computed.  A flat histogram indicates a well-calibrated
    ensemble; a U-shape means under-dispersed; a dome means over-dispersed.

    Reads raw generation files (requires ensemble members, not just metrics).
    Skipped silently for deterministic (single-member) models.
    """
    for spec in specs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            truth_ds, pred_ds, _ = open_samples(spec.path)

        n_ens = pred_ds.sizes.get("ensemble", 1)
        if n_ens <= 1:
            print(f"  Skipping rank histogram for {spec.name}: not an ensemble model.")
            continue

        truth_ds = add_wind_speed(truth_ds)
        pred_ds = add_wind_speed(pred_ds)

        ncols = 3
        nrows = int(np.ceil(len(ALL_VARS) / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axs_flat = np.array(axs).flat

        for v, ax in zip(ALL_VARS, axs_flat):
            truth_arr = truth_ds[v].values   # (time, y, x)
            pred_arr = pred_ds[v].values     # (n_ens, time, y, x)

            # Count how many ensemble members fall below the truth at each point
            ranks = (pred_arr < truth_arr[np.newaxis]).sum(axis=0).ravel()

            ax.hist(ranks, bins=n_ens + 1, range=(-0.5, n_ens + 0.5),
                    density=True, color="#3b7dd8", edgecolor="white", linewidth=0.5)
            ax.axhline(1.0 / (n_ens + 1), color="k", linestyle="--",
                       linewidth=1, label="uniform (ideal)")
            ax.set_title(VAR_LABELS.get(v, v), fontsize=9)
            ax.set_xlabel("Rank")
            ax.set_ylabel("Relative frequency")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

        for ax in list(axs_flat)[len(ALL_VARS):]:
            ax.set_visible(False)

        st = styles[spec.name]
        fig.suptitle(
            f"Rank Histogram (Talagrand) — {st['label']} (n_ens={n_ens})", fontsize=11
        )
        plt.tight_layout()
        _savefig(f"rank_histogram_{spec.name}.png", plots_dir)


# ─── Plot 9 & 10: Spectral / distribution helpers ────────────────────────────

_CHUNK = 50   # timesteps per chunk for large-file plots


def _estimate_dx_km(root) -> float:
    """Estimate mean zonal grid spacing in km from root dataset lat/lon."""
    lon = root["lon"].values
    lat = root["lat"].values
    if lon.ndim == 2:
        dlon = float(np.abs(np.diff(lon[lon.shape[0] // 2, :])).mean())
        lat_c = float(lat[lat.shape[0] // 2, lat.shape[1] // 2])
    else:
        dlon = float(np.abs(np.diff(lon)).mean())
        lat_c = float(np.mean(lat))
    return dlon * 111.0 * np.cos(np.radians(lat_c))


def _zonal_power_spectrum(data, dx_km: float):
    """One-sided zonal PSD averaged over time and rows.

    Args:
        data:   (time, y, x) np.ndarray OR xr.DataArray with a 'time' dim.
                DataArrays are read in chunks to limit memory use.
        dx_km:  grid spacing in km

    Returns:
        freqs: wavenumbers in 1/km  (DC removed)
        psd:   mean PSD in [var²·km]
    """
    if isinstance(data, xr.DataArray):
        n_time = data.sizes["time"]
        n_x = data.shape[-1]
        freqs = np.fft.rfftfreq(n_x, d=dx_km)
        psd_sum = np.zeros(len(freqs))
        n_rows_total = 0
        for t0 in range(0, n_time, _CHUNK):
            arr = data.isel(time=slice(t0, t0 + _CHUNK)).values   # (chunk, y, x)
            fft2d = np.fft.rfft(arr, axis=-1)
            psd_sum += (np.abs(fft2d) ** 2 * dx_km / n_x).sum(axis=(0, 1))
            n_rows_total += arr.shape[0] * arr.shape[1]
        psd = psd_sum / n_rows_total
    else:
        n_time, _n_y, n_x = data.shape
        freqs = np.fft.rfftfreq(n_x, d=dx_km)
        psd_sum = np.zeros(len(freqs))
        for t in range(n_time):
            fft2d = np.fft.rfft(data[t], axis=-1)
            psd_sum += (np.abs(fft2d) ** 2 * dx_km / n_x).mean(axis=0)
        psd = psd_sum / n_time

    psd[1:-1] *= 2    # one-sided: double non-DC/Nyquist
    return freqs[1:], psd[1:]     # drop DC


def _hist_counts(ds: xr.Dataset, var_key: str, bins: np.ndarray) -> np.ndarray:
    """Accumulate histogram counts across time chunks; handles wind_speed_10m."""
    n_time = ds.sizes["time"]
    counts = np.zeros(len(bins) - 1, dtype=np.int64)
    for t0 in range(0, n_time, _CHUNK):
        sl = slice(t0, t0 + _CHUNK)
        if var_key == "wind_speed_10m":
            u = ds["eastward_wind_10m"].isel(time=sl).values
            v = ds["northward_wind_10m"].isel(time=sl).values
            arr = np.sqrt(u ** 2 + v ** 2).ravel()
        else:
            arr = ds[var_key].isel(time=sl).values.ravel()
        c, _ = np.histogram(arr, bins=bins)
        counts += c
    return counts


# ─── Plot 9: Power spectra ────────────────────────────────────────────────────

def plot_power_spectra(specs, styles: dict, plots_dir: str):
    """Zonal power spectra for kinetic energy, temperature, and radar reflectivity."""
    rows = [
        ("kinetic_energy",             "Kinetic energy spectra (m²/s²·km)"),
        ("temperature_2m",             "2m temperature spectra (K²·km)"),
        ("maximum_radar_reflectivity", "Radar reflectivity spectra (dBZ²·km)"),
    ]
    fig, axs = plt.subplots(len(rows), 1, figsize=(8, 4 * len(rows)))
    axs = np.atleast_1d(axs)

    truth_plotted = False

    for spec in specs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            truth_ds, pred_ds, root = open_samples(spec.path)
        dx_km = _estimate_dx_km(root)
        st = styles[spec.name]
        n_ens = pred_ds.sizes.get("ensemble", 1)

        for ax, (var_key, ylabel) in zip(axs, rows):
            if var_key == "kinetic_energy":
                # Pass DataArrays directly — chunked loading inside _zonal_power_spectrum
                freqs, pu_t = _zonal_power_spectrum(truth_ds["eastward_wind_10m"], dx_km)
                _, pv_t = _zonal_power_spectrum(truth_ds["northward_wind_10m"], dx_km)
                truth_psd = pu_t + pv_t

                # Ensemble mean is lazy; each chunk loads and averages on the fly
                _, pu_p = _zonal_power_spectrum(pred_ds["eastward_wind_10m"].mean("ensemble"), dx_km)
                _, pv_p = _zonal_power_spectrum(pred_ds["northward_wind_10m"].mean("ensemble"), dx_km)
                pred_psd = pu_p + pv_p
            else:
                freqs, truth_psd = _zonal_power_spectrum(truth_ds[var_key], dx_km)
                _, pred_psd = _zonal_power_spectrum(pred_ds[var_key].mean("ensemble"), dx_km)

            if not truth_plotted:
                ax.loglog(freqs, truth_psd, color="gold", linewidth=2, label="truth", zorder=3)
            ax.loglog(freqs, pred_psd, color=st["color"], label=st["label"],
                      linestyle="--" if n_ens > 1 else "-", linewidth=1.5)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_xlabel("Zonal wavenumber (1/km)", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3, which="both")

        truth_plotted = True

    fig.suptitle("Zonal power spectra", fontsize=11)
    plt.tight_layout()
    _savefig("power_spectra.png", plots_dir)


# ─── Plot 10: Log-PDF distributions ──────────────────────────────────────────

def plot_distributions(specs, styles: dict, plots_dir: str):
    """Log-PDF distributions for wind speed, temperature, and radar reflectivity."""
    dist_vars = [
        ("wind_speed_10m",             "10m wind speed (m/s)"),
        ("temperature_2m",             "2m temperature (K)"),
        ("maximum_radar_reflectivity", "Radar reflectivity (dBZ)"),
    ]
    fig, axs = plt.subplots(1, len(dist_vars), figsize=(5 * len(dist_vars), 5))
    axs = np.atleast_1d(axs)

    truth_plotted = False

    for spec in specs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            truth_ds, pred_ds, _ = open_samples(spec.path)
        st = styles[spec.name]
        n_ens = pred_ds.sizes.get("ensemble", 1)

        for ax, (var_key, xlabel) in zip(axs, dist_vars):
            # Estimate bin range from a downsampled truth sample (avoid loading all)
            step = max(1, truth_ds.sizes["time"] // 50)
            if var_key == "wind_speed_10m":
                u = truth_ds["eastward_wind_10m"].isel(time=slice(None, None, step)).values
                v = truth_ds["northward_wind_10m"].isel(time=slice(None, None, step)).values
                sample = np.sqrt(u ** 2 + v ** 2).ravel()
            else:
                sample = truth_ds[var_key].isel(time=slice(None, None, step)).values.ravel()
            vmin = np.percentile(sample, 0.1)
            vmax = np.percentile(sample, 99.9)
            bins = np.linspace(vmin, vmax, 100)
            centers = 0.5 * (bins[:-1] + bins[1:])
            widths = np.diff(bins)

            if not truth_plotted:
                counts_t = _hist_counts(truth_ds, var_key, bins)
                density_t = counts_t / (counts_t.sum() * widths)
                with np.errstate(divide="ignore"):
                    log_pdf = np.log(np.where(density_t > 0, density_t, np.nan))
                ax.plot(centers, log_pdf, color="gold", linewidth=2, label="truth", zorder=3)
                ax.set_xlabel(xlabel, fontsize=9)
                ax.set_ylabel("log(PDF)", fontsize=9)
                ax.grid(alpha=0.3)

            counts_p = _hist_counts(pred_ds, var_key, bins)
            density_p = counts_p / (counts_p.sum() * widths)
            with np.errstate(divide="ignore"):
                log_pdf_p = np.log(np.where(density_p > 0, density_p, np.nan))
            ax.plot(centers, log_pdf_p, color=st["color"], label=st["label"],
                    linestyle="--" if n_ens > 1 else "-", linewidth=1.5)
            ax.legend(fontsize=8)

        truth_plotted = True

    fig.suptitle("Log-PDF distributions", fontsize=11)
    plt.tight_layout()
    _savefig("distributions.png", plots_dir)


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
    metrics_dict = {spec.name: load_metrics(spec, scores_dir, len(specs)) for spec in specs}

    print("Plot 1: RMSE time series")
    plot_metric_timeseries("rmse", specs, metrics_dict, styles, plots_dir)

    print("Plot 2: CRPS time series")
    plot_metric_timeseries("crps", specs, metrics_dict, styles, plots_dir)

    print("Plot 3: Bias maps")
    plot_spatial_map("bias", specs, metrics_dict, styles, plots_dir)

    print("Plot 4: Error maps")
    plot_spatial_map("error", specs, metrics_dict, styles, plots_dir)

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

    print("Plot 8: Rank histograms")
    plot_rank_histogram(specs, styles, plots_dir)

    print("Plot 9: Power spectra")
    plot_power_spectra(specs, styles, plots_dir)

    print("Plot 10: Log-PDF distributions")
    plot_distributions(specs, styles, plots_dir)

    print(f"\nAll plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
