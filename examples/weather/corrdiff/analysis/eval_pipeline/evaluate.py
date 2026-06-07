"""Single-file evaluation driver.

Streams a generation ``.nc`` file once, feeding every timestep into per-variable
accumulators from :mod:`analysis.eval_pipeline.core`, then emits metrics (JSON + CSV)
and diagnostic plots. The accumulators are also returned so the two-file comparison
layer can build overlays from the same numbers.
"""

from __future__ import annotations

import dataclasses
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.eval_pipeline.core.metrics import MetricsAccumulator
from analysis.eval_pipeline.core.plots import (
    HistogramAccumulator,
    RAPSDAccumulator,
    plot_diagnostic_panel,
)

from analysis.eval_pipeline.io import Channel, GenerationFile, channels_by_name

# Scalar metrics tabulated in metrics.csv / the comparison table, in display order.
SCALAR_METRICS = ["rmse", "mae", "bias", "pcc", "crps", "spread", "skill", "spread_skill_ratio"]


@dataclasses.dataclass
class ChannelAcc:
    """The three trusted accumulators for one variable."""
    metrics: MetricsAccumulator
    hist: HistogramAccumulator
    rapsd: RAPSDAccumulator


@dataclasses.dataclass
class EvalResult:
    name: str
    kind: str                       # "diffusion" or "regression"
    is_ensemble: bool
    channels: list[Channel]
    accs: dict[str, ChannelAcc]     # keyed by channel name
    metrics: dict                   # flat, prefixed "var/metric"
    table: pd.DataFrame             # index=variable, cols=SCALAR_METRICS
    gf: GenerationFile

    def close(self):
        self.gf.close()


def resolve_kind(gf: GenerationFile, kind: str) -> tuple[str, bool]:
    """Resolve --kind {auto,diffusion,regression} into (kind, is_ensemble)."""
    if kind == "auto":
        is_ens = gf.is_ensemble
        return ("diffusion" if is_ens else "regression"), is_ens
    if kind == "regression":
        return "regression", False
    return "diffusion", gf.is_ensemble


def run_accumulators(
    gf: GenerationFile, channels: list[Channel], is_ensemble: bool, verbose: bool = True
) -> dict[str, ChannelAcc]:
    """Single streaming pass: update per-variable accumulators for every timestep."""
    H, W = gf.img_shape
    dx_km = gf.dx_km()
    accs = {
        c.name: ChannelAcc(
            # CWB variables are not precipitation-in-mm: skip_conditional_metrics
            # disables the mm thresholds AND the clamp(min=0) that would corrupt
            # signed wind / temperature. skip_spread_skill for deterministic models.
            metrics=MetricsAccumulator(
                skip_conditional_metrics=True,
                skip_spread_skill=not is_ensemble,
            ),
            hist=HistogramAccumulator(),
            rapsd=RAPSDAccumulator(img_shape=(H, W), dx_km=dx_km),
        )
        for c in channels
    }

    for t, _time, pred_ens, target in gf.iter_timesteps(channels):
        for ci, c in enumerate(channels):
            p = pred_ens[:, ci : ci + 1]      # (N_ens, 1, H, W)
            y = target[ci : ci + 1]           # (1, H, W)
            acc = accs[c.name]
            acc.metrics.update(p, y)
            acc.hist.update(p, y)
            acc.rapsd.update(p, y)
        if verbose and (t + 1) % 25 == 0:
            print(f"    ...{t + 1}/{gf.n_time} timesteps")
    return accs


def collect_metrics(accs: dict[str, ChannelAcc], channels: list[Channel]) -> tuple[dict, pd.DataFrame]:
    """Build the flat metrics dict (prefixed) and a tidy per-variable scalar table."""
    flat: dict = {}
    rows = {}
    for c in channels:
        d = accs[c.name].metrics.to_dict(prefix=f"{c.name}/")
        flat.update(d)
        rows[c.name] = {m: d.get(f"{c.name}/{m}", np.nan) for m in SCALAR_METRICS}
    table = pd.DataFrame.from_dict(rows, orient="index")[SCALAR_METRICS]
    table.index.name = "variable"
    return flat, table


def evaluate_file(
    path: str, name: str, kind: str = "auto", channel_names: list[str] | None = None,
    verbose: bool = True,
) -> EvalResult:
    """Run the full accumulator pass over one file and assemble an EvalResult."""
    channels = channels_by_name(channel_names)
    gf = GenerationFile(path, channels)
    kind, is_ens = resolve_kind(gf, kind)
    if verbose:
        tag = f"ensemble (N={gf.n_ensemble})" if is_ens else "deterministic"
        print(f"  [{name}] {gf.stem}: {gf.n_time} timesteps, {tag}, kind={kind}")
    accs = run_accumulators(gf, channels, is_ens, verbose=verbose)
    flat, table = collect_metrics(accs, channels)
    return EvalResult(name, kind, is_ens, channels, accs, flat, table, gf)


# ── Output ───────────────────────────────────────────────────────────────────


def _save_fig(fig, plots_dir: str, fname: str, wb=None) -> str:
    os.makedirs(plots_dir, exist_ok=True)
    out = os.path.join(plots_dir, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    if wb is not None:
        wb.log_figure(os.path.splitext(fname)[0], fig)
    plt.close(fig)
    print(f"    saved {out}")
    return out


def _event_indices(gf: GenerationFile, n_samples: int, event_times: list[str] | None) -> list[int]:
    """Chronologically-ordered file indices: forced event times first, then evenly filled."""
    order = gf.order()
    times = pd.to_datetime(gf.times[order])
    forced: list[int] = []
    for ts in event_times or []:
        day = pd.Timestamp(ts).normalize()
        match = np.where(times.normalize() == day)[0]
        if len(match):
            forced.append(int(match[0]))
        else:
            print(f"    WARNING: event {ts} not in {gf.stem}, skipping")
    remaining = max(n_samples - len(forced), 0)
    fill = [i for i in np.linspace(0, len(times) - 1, remaining, dtype=int) if i not in forced]
    sorted_idx = sorted(set(forced + fill))
    return [int(order[i]) for i in sorted_idx]


def plot_event_panel(gf: GenerationFile, channels: list[Channel], file_idx: int, wb=None,
                     plots_dir: str = "."):
    """Spatial panel for one timestep: rows=variables, cols=[Truth | Ensemble mean | Spread]."""
    pred, target = gf.event_fields(file_idx, channels)   # (N,C,H,W),(C,H,W)
    ens_mean = pred.mean(axis=0)
    ens_spread = pred.std(axis=0)
    n = len(channels)
    has_spread = pred.shape[0] > 1
    ncols = 3 if has_spread else 2
    fig, axs = plt.subplots(n, ncols, figsize=(5 * ncols, 4 * n), squeeze=False)
    lon, lat = gf.lon2d, gf.lat2d
    col_titles = ["Truth", "Ensemble mean"] + (["Ensemble spread"] if has_spread else [])
    for j, ct in enumerate(col_titles):
        axs[0, j].set_title(ct, fontsize=10)
    for i, c in enumerate(channels):
        cmap = "RdBu_r" if c.signed else "viridis"
        vmin = min(target[i].min(), ens_mean[i].min())
        vmax = max(target[i].max(), ens_mean[i].max())
        if c.signed:
            b = max(abs(vmin), abs(vmax)); vmin, vmax = -b, b
        axs[i, 0].pcolormesh(lon, lat, target[i], cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        axs[i, 0].set_ylabel(f"{c.label} [{c.unit}]", fontsize=8)
        im = axs[i, 1].pcolormesh(lon, lat, ens_mean[i], cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        plt.colorbar(im, ax=axs[i, 1], fraction=0.046)
        if has_spread:
            im2 = axs[i, 2].pcolormesh(lon, lat, ens_spread[i], cmap="YlOrRd", shading="auto")
            plt.colorbar(im2, ax=axs[i, 2], fraction=0.046)
    tstr = pd.Timestamp(gf.times[file_idx]).strftime("%Y-%m-%d %H:%M UTC")
    fig.suptitle(f"{gf.stem} — {tstr}", fontsize=11)
    fig.tight_layout()
    _save_fig(fig, plots_dir, f"event_{pd.Timestamp(gf.times[file_idx]).strftime('%Y%m%d_%H%M')}.png", wb)


def write_outputs(result: EvalResult, outdir: str, n_samples: int = 3,
                  event_times: list[str] | None = None, wb=None):
    """Write metrics (JSON+CSV), per-variable diagnostic panels, and event panels."""
    os.makedirs(outdir, exist_ok=True)
    plots_dir = os.path.join(outdir, "plots")

    # Metrics
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(result.metrics, f, indent=2, default=float)
    result.table.to_csv(os.path.join(outdir, "metrics.csv"))
    print(f"  metrics → {outdir}/metrics.{{json,csv}}")
    print(result.table.round(4).to_string())
    if wb is not None:
        wb.log_metrics(result.metrics)
        wb.log_table(result.table.reset_index(), name=f"{result.name}/metrics")

    # Per-variable diagnostic panels (spread-skill, rank hist, Q-Q, hist, RAPSD, log-PDF).
    dx_km = result.gf.dx_km()
    for c in result.channels:
        acc = result.accs[c.name]
        md = acc.metrics.to_dict(prefix=f"{c.name}/")
        fig = plot_diagnostic_panel(md, c.name, acc.hist, acc.rapsd, rapsd_dx_km=dx_km,
                                    diagnostic_info=c.diagnostic_info)
        if fig is not None:
            _save_fig(fig, plots_dir, f"diagnostics_{c.name}.png", wb)

    # Event panels
    for fi in _event_indices(result.gf, n_samples, event_times):
        plot_event_panel(result.gf, result.channels, fi, wb=wb, plots_dir=plots_dir)


def run_single(path: str, name: str, kind: str, outdir_base: str, channel_names=None,
               n_samples: int = 3, event_times=None, wb=None) -> EvalResult:
    """End-to-end single-file evaluation: compute, write, return the result."""
    result = evaluate_file(path, name, kind, channel_names)
    outdir = os.path.join(outdir_base, result.gf.stem)
    write_outputs(result, outdir, n_samples=n_samples, event_times=event_times, wb=wb)
    return result
