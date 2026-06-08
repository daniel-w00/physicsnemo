"""Multi-file comparison layer.

Evaluates two or more generation files with the single-file driver (so the underlying
numbers come from the core accumulators), then builds overlay plots and a side-by-side
metric table that the single-model core plots can't produce alone.
Handles regression (deterministic) vs diffusion (ensemble): ensemble-only quantities are
marked N/A for deterministic models and ensemble-only overlays are skipped.

The first model in the list is the *reference*: truth, the grid, and the event-timestep
selection are taken from it, and the other models are aligned to its timestamps.
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

# Distinct, print-friendly colours; falls back to tab20 beyond this many models.
_PALETTE = [
    "#3b7dd8", "#e07b39", "#3fa34d", "#c0392b", "#8e44ad",
    "#16a085", "#d4ac0d", "#7f8c8d", "#2c3e50", "#e84393",
]
_TRUTH_COLOR = "#444444"


def _colors(n: int) -> list:
    """Return n distinct colours."""
    if n <= len(_PALETTE):
        return _PALETTE[:n]
    cmap = plt.get_cmap("tab20")
    return [cmap(i % 20) for i in range(n)]


def _dedupe_names(results: list[EvalResult]) -> None:
    """Make model names unique in place (table columns / legends rely on this)."""
    seen: dict[str, int] = {}
    for r in results:
        if r.name in seen:
            seen[r.name] += 1
            r.name = f"{r.name}#{seen[r.name]}"
        else:
            seen[r.name] = 0


def _grid(n: int):
    ncols = 3 if n > 4 else min(n, 2)
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4 * nrows), squeeze=False)
    axs_flat = axs.flat
    return fig, list(axs_flat)


def _hide_unused(axs_flat, n):
    for ax in axs_flat[n:]:
        ax.set_visible(False)


def _title(results: list[EvalResult]) -> str:
    return " vs ".join(r.name for r in results)


# ── Comparison table ─────────────────────────────────────────────────────────


def comparison_table(results: list[EvalResult]) -> pd.DataFrame:
    """MultiIndex-column table: per variable × metric → one column per model."""
    names = [r.name for r in results]
    cols = pd.MultiIndex.from_product([SCALAR_METRICS, names], names=["metric", "model"])
    # Union of variables, preserving the reference model's order first.
    all_vars: list[str] = list(results[0].table.index)
    for r in results[1:]:
        for v in r.table.index:
            if v not in all_vars:
                all_vars.append(v)
    data = {}
    for var in all_vars:
        row = {}
        for m in SCALAR_METRICS:
            for r in results:
                row[(m, r.name)] = r.table.loc[var, m] if var in r.table.index else np.nan
        data[var] = row
    df = pd.DataFrame.from_dict(data, orient="index").reindex(columns=cols)
    df.index.name = "variable"
    return df


# ── Overlay plots ────────────────────────────────────────────────────────────


def plot_rapsd_overlay(results: list[EvalResult], channels: list[Channel]):
    """RAPSD per variable: truth (from reference) + every model."""
    colors = _colors(len(results))
    fig, axs = _grid(len(channels))
    for ax, c in zip(axs, channels):
        for res, col in zip(results, colors):
            acc = res.accs[c.name].rapsd
            if acc.n_samples == 0:
                continue
            freq = np.asarray(acc.bin_centers, float)
            pred = (acc.pred_psd_sum / acc.n_samples).numpy()
            v = (freq > 0) & (pred > 0)
            ax.loglog(freq[v], pred[v], color=col, lw=1.6, label=res.name)
        acc_t = results[0].accs[c.name].rapsd
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
    fig.suptitle(f"RAPSD — {_title(results)}", fontsize=12)
    fig.tight_layout()
    return fig


def plot_pdf_overlay(results: list[EvalResult], channels: list[Channel]):
    """Log-scale distribution per variable: truth (from reference) + every model's ens-avg."""
    colors = _colors(len(results))
    fig, axs = _grid(len(channels))
    for ax, c in zip(axs, channels):
        ref = results[0].accs[c.name].hist.get_rebinned()
        ax.plot(ref["bin_centers"], ref["target"], color=_TRUTH_COLOR, lw=2, label="truth", zorder=3)
        for res, col in zip(results, colors):
            d = res.accs[c.name].hist.get_rebinned()
            ax.plot(d["bin_centers"], d["ens_avg_hist"], color=col, lw=1.5, label=res.name)
        ax.set_yscale("log")
        ax.set_ylim(bottom=1.0)
        ax.set_title(c.label, fontsize=10)
        ax.set_xlabel(f"{c.label} [{c.unit}]", fontsize=8)
        ax.set_ylabel("frequency", fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    _hide_unused(axs, len(channels))
    fig.suptitle(f"Distributions — {_title(results)}", fontsize=12)
    fig.tight_layout()
    return fig


def plot_spread_skill_overlay(results: list[EvalResult], channels: list[Channel]):
    """Spread-skill reliability overlay; only ensemble models are drawn."""
    if not any(r.is_ensemble for r in results):
        return None
    colors = _colors(len(results))
    fig, axs = _grid(len(channels))
    for ax, c in zip(axs, channels):
        gmax = 0.0
        for res, col in zip(results, colors):
            if not res.is_ensemble:
                continue
            md = res.accs[c.name].metrics.to_dict(prefix=f"{c.name}/")
            sp = np.asarray(md.get(f"{c.name}/spread_skill_bin_mean_spread", []), float)
            sk = np.asarray(md.get(f"{c.name}/spread_skill_bin_mean_skill", []), float)
            valid = (sp > 0) | (sk > 0)
            sp, sk = sp[valid], sk[valid]
            if sp.size:
                ax.scatter(sk, sp, s=40, color=col, label=res.name, zorder=3)
                gmax = max(gmax, sp.max(), sk.max())
        if gmax > 0:
            ax.plot([0, gmax * 1.05], [0, gmax * 1.05], "k--", lw=1, label="y=x")
        ax.set_title(c.label, fontsize=10)
        ax.set_xlabel("RMSE", fontsize=8)
        ax.set_ylabel("spread", fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    _hide_unused(axs, len(channels))
    fig.suptitle(f"Spread-skill — {_title(results)}", fontsize=12)
    fig.tight_layout()
    return fig


def plot_metric_bars(results: list[EvalResult], channels: list[Channel],
                     metrics=("rmse", "mae", "crps")):
    """Grouped bar chart of time-mean metrics per variable, one bar group per model."""
    colors = _colors(len(results))
    names = [c.name for c in channels]
    labels = [c.label for c in channels]
    x = np.arange(len(channels))
    n = len(results)
    width = 0.8 / n
    fig, axs = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.5), squeeze=False)
    for j, m in enumerate(metrics):
        ax = axs[0, j]
        for k, (res, col) in enumerate(zip(results, colors)):
            vals = [res.table.loc[nm, m] if nm in res.table.index else np.nan for nm in names]
            offset = (k - (n - 1) / 2) * width
            ax.bar(x + offset, vals, width, color=col, label=res.name)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_title(m.upper(), fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Time-mean metrics — {_title(results)}", fontsize=12)
    fig.tight_layout()
    return fig


def plot_spatial_compare(results: list[EvalResult], channels: list[Channel], ref_idx: int):
    """Spatial maps at one timestep: rows=variables, cols=[truth | each model's mean].

    The reference (results[0]) defines the timestamp; models without a matching
    timestamp are dropped from the figure for this event.
    """
    ref = results[0]
    time_ref = ref.gf.times[ref_idx]
    means: list[tuple[str, np.ndarray]] = []
    target = None
    for res in results:
        if res is ref:
            idx = ref_idx
        else:
            match = np.where(res.gf.times == time_ref)[0]
            if not len(match):
                continue
            idx = int(match[0])
        pred, tgt = res.gf.event_fields(idx, channels)
        if target is None:
            target = tgt
        means.append((res.name, pred.mean(axis=0)))

    lon, lat = ref.gf.lon2d, ref.gf.lat2d
    n = len(channels)
    ncols = 1 + len(means)
    fig, axs = plt.subplots(n, ncols, figsize=(5 * ncols, 4 * n), squeeze=False)
    titles = ["Truth"] + [nm for nm, _ in means]
    for j, ti in enumerate(titles):
        axs[0, j].set_title(ti, fontsize=10)
    for i, c in enumerate(channels):
        cmap = "RdBu_r" if c.signed else "viridis"
        fields = [target[i]] + [mean[i] for _, mean in means]
        vmin = min(f.min() for f in fields)
        vmax = max(f.max() for f in fields)
        if c.signed:
            bnd = max(abs(vmin), abs(vmax)); vmin, vmax = -bnd, bnd
        for j, f in enumerate(fields):
            im = axs[i, j].pcolormesh(lon, lat, f, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        plt.colorbar(im, ax=axs[i, ncols - 1], fraction=0.046)
        axs[i, 0].set_ylabel(f"{c.label} [{c.unit}]", fontsize=8)
    tstr = pd.Timestamp(time_ref).strftime("%Y-%m-%d %H:%M UTC")
    fig.suptitle(f"{_title(results)} — {tstr}", fontsize=11)
    fig.tight_layout()
    return fig


# ── Orchestration ────────────────────────────────────────────────────────────


def _outdir_tag(stems: list[str]) -> str:
    """Compact, filesystem-safe directory name for a comparison run."""
    tag = "_vs_".join(stems)
    if len(tag) <= 100:
        return tag
    return f"compare_{len(stems)}models_{stems[0][:40]}"


def run_compare(models, outdir_base, channel_names=None, n_samples=3,
                event_times=None, wb=None) -> list[EvalResult]:
    """Evaluate and compare 2+ generation files.

    ``models`` is a list of dicts with keys ``pred`` (path), ``name``, and ``kind``
    (``auto``/``diffusion``/``regression``). The first entry is the reference.
    """
    results: list[EvalResult] = []
    for m in models:
        print(f"Evaluating {m['name']}")
        results.append(evaluate_file(m["pred"], m["name"], m.get("kind") or "auto", channel_names))
    _dedupe_names(results)
    channels = results[0].channels

    stems = [r.gf.stem for r in results]
    outdir = os.path.join(outdir_base, _outdir_tag(stems))
    plots_dir = os.path.join(outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Comparison table
    table = comparison_table(results)
    table.to_csv(os.path.join(outdir, "metrics_comparison.csv"))
    print(f"\nComparison ({_title(results)}):")
    print(table.round(4).to_string())
    print(f"  full table → {outdir}/metrics_comparison.csv")
    if wb is not None:
        flat = table.copy()
        flat.columns = [f"{m}/{mod}" for m, mod in flat.columns]
        wb.log_table(flat.reset_index(), name="metrics_comparison")

    # Overlay plots
    _save_fig(plot_rapsd_overlay(results, channels), plots_dir, "compare_rapsd.png", wb)
    _save_fig(plot_pdf_overlay(results, channels), plots_dir, "compare_distributions.png", wb)
    ss = plot_spread_skill_overlay(results, channels)
    if ss is not None:
        _save_fig(ss, plots_dir, "compare_spread_skill.png", wb)
    _save_fig(plot_metric_bars(results, channels), plots_dir, "compare_metric_bars.png", wb)

    # Spatial sample maps at selected events (indices from the reference model)
    for fi in _event_indices(results[0].gf, n_samples, event_times):
        fig = plot_spatial_compare(results, channels, fi)
        stamp = pd.Timestamp(results[0].gf.times[fi]).strftime("%Y%m%d_%H%M")
        _save_fig(fig, plots_dir, f"compare_spatial_{stamp}.png", wb)

    return results
