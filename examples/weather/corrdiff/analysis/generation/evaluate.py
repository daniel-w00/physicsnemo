"""Evaluate CorrDiff generation outputs: compute metrics and generate all plots.

Run from the repo root:

    # Single model
    python analysis/generation/evaluate.py --model "regression:output/gen_taiwan/reg.nc"

    # Two-model comparison
    python analysis/generation/evaluate.py \\
        --model "diffusion:output/gen_taiwan/diff.nc" \\
        --model "regression:output/gen_taiwan/reg.nc" \\
        --event "2021-09-12:Typhoon Chanthu"
"""

import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.generation.utils import assign_styles, make_output_dir, parse_model_args
from analysis.generation.metrics import compute_metrics_for_file
from analysis.generation.plots import (
    plot_metric_timeseries,
    plot_spatial_map,
    plot_spread_skill,
    plot_sample_panels,
    plot_summary_bars,
    plot_rank_histogram,
    plot_power_spectra,
    plot_distributions,
)


def main():
    parser = argparse.ArgumentParser(
        description="Compute metrics and generate plots for CorrDiff generation outputs."
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
    os.makedirs(scores_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics_dict = {}
    for spec in specs:
        print(f"Computing metrics: {spec.display_name}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = compute_metrics_for_file(spec.path, spec.name)
        ds.attrs["checkpoint"] = spec.ckpt
        nc_path = os.path.join(scores_dir, "metrics.nc" if len(specs) == 1 else f"{spec.name}_metrics.nc")
        ds.to_netcdf(nc_path)
        print(f"  Saved: {nc_path}")
        metrics_dict[spec.name] = ds

    # ── Plots ─────────────────────────────────────────────────────────────────
    styles = assign_styles(specs)

    events = {}
    for e in args.event:
        parts = e.split(":", maxsplit=1)
        if len(parts) != 2:
            print(f"  WARNING: --event must be DATE:LABEL, got {e!r}, skipping.")
            continue
        events[parts[0].strip()] = parts[1].strip()

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

    print(f"\nDone. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
