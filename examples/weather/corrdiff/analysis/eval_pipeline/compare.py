"""Two-file comparison layer.

Evaluates two generation files with the single-file driver (so the underlying numbers
come from the core accumulators), then builds the overlay plots and the side-by-side
metric table that the single-model core plots can't produce alone.
Handles regression (deterministic) vs diffusion (ensemble): ensemble-only quantities are
marked N/A for deterministic models and ensemble-only overlays are skipped.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.eval_pipeline.evaluate import (
    SCALAR_METRICS,
    EvalResult,
    _event_indices,
    _save_fig,
    evaluate_file,
)
from analysis.eval_pipeline.io import Channel

_COLORS = {"a": "#3b7dd8", "b": "#e07b39"}
_TRUTH_COLOR = "#444444"


def _grid(n: int):
    ncols = 3 if n > 4 else min(n, 2)
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4 * nrows), squeeze=False)
    axs_flat = axs.flat
    return fig, list(axs_flat)


def _hide_unused(axs_flat, n):
    for ax in axs_flat[n:]:
        ax.set_visible(False)


# ── Comparison table ─────────────────────────────────────────────────────────


def comparison_table(ra: EvalResult, rb: EvalResult) -> pd.DataFrame:
    """MultiIndex-column table: per variable × metric → [name_a, name_b, delta]."""
    cols = pd.MultiIndex.from_product(
        [SCALAR_METRICS, [ra.name, rb.name, "delta"]], names=["metric", "model"]
    )
    data = {}
    for var in ra.table.index:
        row = {}
        for m in SCALAR_METRICS:
            a = ra.table.loc[var, m]
            b = rb.table.loc[var, m] if var in rb.table.index else np.nan
            row[(m, ra.name)] = a
            row[(m, rb.name)] = b
            row[(m, "delta")] = a - b
        data[var] = row
    df = pd.DataFrame.from_dict(data, orient="index")
    df = df.reindex(columns=cols)
    df.index.name = "variable"
    return df


# ── Overlay plots ────────────────────────────────────────────────────────────


def plot_rapsd_overlay(ra: EvalResult, rb: EvalResult, channels: list[Channel]):
    """RAPSD per variable: truth (from A) + model A + model B."""
    fig, axs = _grid(len(channels))
    for ax, c in zip(axs, channels):
        for res, key in ((ra, "a"), (rb, "b")):
            acc = res.accs[c.name].rapsd
            if acc.n_samples == 0:
                continue
            freq = np.asarray(acc.bin_centers, float)
            pred = (acc.pred_psd_sum / acc.n_samples).numpy()
            v = (freq > 0) & (pred > 0)
            ax.loglog(freq[v], pred[v], color=_COLORS[key], lw=1.6, label=res.name)
        acc_t = ra.accs[c.name].rapsd
        tgt = (acc_t.target_psd_sum / max(acc_t.n_samples, 1)).numpy()
        freq = np.asarray(acc_t.bin_centers, float)
        v = (freq > 0) & (tgt > 0)
        ax.loglog(freq[v], tgt[v], color=_TRUTH_COLOR, lw=2, label="truth", zorder=3)
        ax.set_title(c.label, fontsize=10)
        ax.set_xlabel("spatial freq (1/km)", fontsize=8)
        ax.set_ylabel("PSD", fontsize=8)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    _hide_unused(axs, len(channels))
    fig.suptitle(f"RAPSD — {ra.name} vs {rb.name}", fontsize=12)
    fig.tight_layout()
    return fig


def plot_pdf_overlay(ra: EvalResult, rb: EvalResult, channels: list[Channel]):
    """Log-scale distribution per variable: truth (from A) + A ens-avg + B ens-avg."""
    fig, axs = _grid(len(channels))
    for ax, c in zip(axs, channels):
        ref = ra.accs[c.name].hist.get_rebinned()
        ax.plot(ref["bin_centers"], ref["target"], color=_TRUTH_COLOR, lw=2, label="truth", zorder=3)
        for res, key in ((ra, "a"), (rb, "b")):
            d = res.accs[c.name].hist.get_rebinned()
            ax.plot(d["bin_centers"], d["ens_avg_hist"], color=_COLORS[key], lw=1.5, label=res.name)
        ax.set_yscale("log")
        ax.set_ylim(bottom=1.0)
        ax.set_title(c.label, fontsize=10)
        ax.set_xlabel(f"{c.label} [{c.unit}]", fontsize=8)
        ax.set_ylabel("frequency", fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    _hide_unused(axs, len(channels))
    fig.suptitle(f"Distributions — {ra.name} vs {rb.name}", fontsize=12)
    fig.tight_layout()
    return fig


def plot_spread_skill_overlay(ra: EvalResult, rb: EvalResult, channels: list[Channel]):
    """Spread-skill reliability overlay; only for ensemble vs ensemble."""
    ens = [r for r in (ra, rb) if r.is_ensemble]
    if len(ens) < 1:
        return None
    fig, axs = _grid(len(channels))
    for ax, c in zip(axs, channels):
        gmax = 0.0
        for res, key in ((ra, "a"), (rb, "b")):
            if not res.is_ensemble:
                continue
            md = res.accs[c.name].metrics.to_dict(prefix=f"{c.name}/")
            sp = np.asarray(md.get(f"{c.name}/spread_skill_bin_mean_spread", []), float)
            sk = np.asarray(md.get(f"{c.name}/spread_skill_bin_mean_skill", []), float)
            valid = (sp > 0) | (sk > 0)
            sp, sk = sp[valid], sk[valid]
            if sp.size:
                ax.scatter(sk, sp, s=40, color=_COLORS[key], label=res.name, zorder=3)
                gmax = max(gmax, sp.max(), sk.max())
        if gmax > 0:
            ax.plot([0, gmax * 1.05], [0, gmax * 1.05], "k--", lw=1, label="y=x")
        ax.set_title(c.label, fontsize=10)
        ax.set_xlabel("RMSE", fontsize=8)
        ax.set_ylabel("spread", fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    _hide_unused(axs, len(channels))
    fig.suptitle(f"Spread-skill — {ra.name} vs {rb.name}", fontsize=12)
    fig.tight_layout()
    return fig


def plot_metric_bars(ra: EvalResult, rb: EvalResult, channels: list[Channel],
                     metrics=("rmse", "mae", "crps")):
    """Grouped bar chart of time-mean metrics per variable, A vs B."""
    names = [c.name for c in channels]
    labels = [c.label for c in channels]
    x = np.arange(len(channels))
    fig, axs = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.5), squeeze=False)
    for j, m in enumerate(metrics):
        ax = axs[0, j]
        a = [ra.table.loc[n, m] for n in names]
        b = [rb.table.loc[n, m] if n in rb.table.index else np.nan for n in names]
        ax.bar(x - 0.2, a, 0.4, color=_COLORS["a"], label=ra.name)
        ax.bar(x + 0.2, b, 0.4, color=_COLORS["b"], label=rb.name)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_title(m.upper(), fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Time-mean metrics — {ra.name} vs {rb.name}", fontsize=12)
    fig.tight_layout()
    return fig


def plot_spatial_compare(ra: EvalResult, rb: EvalResult, channels: list[Channel], file_idx_a: int):
    """Spatial maps at one timestep: rows=variables, cols=[truth | A mean | B mean]."""
    time_a = ra.gf.times[file_idx_a]
    match = np.where(rb.gf.times == time_a)[0]
    pred_a, target = ra.gf.event_fields(file_idx_a, channels)
    a_mean = pred_a.mean(axis=0)
    if len(match):
        pred_b, _ = rb.gf.event_fields(int(match[0]), channels)
        b_mean = pred_b.mean(axis=0)
        ncols = 3
    else:
        b_mean = None
        ncols = 2
    lon, lat = ra.gf.lon2d, ra.gf.lat2d
    n = len(channels)
    fig, axs = plt.subplots(n, ncols, figsize=(5 * ncols, 4 * n), squeeze=False)
    titles = ["Truth", ra.name] + ([rb.name] if b_mean is not None else [])
    for j, ti in enumerate(titles):
        axs[0, j].set_title(ti, fontsize=10)
    for i, c in enumerate(channels):
        cmap = "RdBu_r" if c.signed else "viridis"
        fields = [target[i], a_mean[i]] + ([b_mean[i]] if b_mean is not None else [])
        vmin = min(f.min() for f in fields)
        vmax = max(f.max() for f in fields)
        if c.signed:
            bnd = max(abs(vmin), abs(vmax)); vmin, vmax = -bnd, bnd
        for j, f in enumerate(fields):
            im = axs[i, j].pcolormesh(lon, lat, f, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        plt.colorbar(im, ax=axs[i, ncols - 1], fraction=0.046)
        axs[i, 0].set_ylabel(f"{c.label} [{c.unit}]", fontsize=8)
    tstr = pd.Timestamp(time_a).strftime("%Y-%m-%d %H:%M UTC")
    fig.suptitle(f"{ra.name} vs {rb.name} — {tstr}", fontsize=11)
    fig.tight_layout()
    return fig


# ── Orchestration ────────────────────────────────────────────────────────────


def run_compare(path_a, name_a, kind_a, path_b, name_b, kind_b, outdir_base,
                channel_names=None, n_samples=3, event_times=None, wb=None) -> tuple[EvalResult, EvalResult]:
    print(f"Evaluating A: {name_a}")
    ra = evaluate_file(path_a, name_a, kind_a, channel_names)
    print(f"Evaluating B: {name_b}")
    rb = evaluate_file(path_b, name_b, kind_b, channel_names)
    channels = ra.channels

    outdir = os.path.join(outdir_base, f"{ra.gf.stem}_vs_{rb.gf.stem}")
    plots_dir = os.path.join(outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Comparison table
    table = comparison_table(ra, rb)
    table.to_csv(os.path.join(outdir, "metrics_comparison.csv"))
    print(f"\nComparison ({ra.name} vs {rb.name}):")
    print(table.xs(ra.name, axis=1, level="model").round(4).to_string())
    print(f"  full table → {outdir}/metrics_comparison.csv")
    if wb is not None:
        flat = table.copy()
        flat.columns = [f"{m}/{mod}" for m, mod in flat.columns]
        wb.log_table(flat.reset_index(), name="metrics_comparison")

    # Overlay plots
    _save_fig(plot_rapsd_overlay(ra, rb, channels), plots_dir, "compare_rapsd.png", wb)
    _save_fig(plot_pdf_overlay(ra, rb, channels), plots_dir, "compare_distributions.png", wb)
    ss = plot_spread_skill_overlay(ra, rb, channels)
    if ss is not None:
        _save_fig(ss, plots_dir, "compare_spread_skill.png", wb)
    _save_fig(plot_metric_bars(ra, rb, channels), plots_dir, "compare_metric_bars.png", wb)

    # Spatial sample maps at selected events (indices from A)
    for fi in _event_indices(ra.gf, n_samples, event_times):
        fig = plot_spatial_compare(ra, rb, channels, fi)
        stamp = pd.Timestamp(ra.gf.times[fi]).strftime("%Y%m%d_%H%M")
        _save_fig(fig, plots_dir, f"compare_spatial_{stamp}.png", wb)

    return ra, rb
