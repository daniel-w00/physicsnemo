"""CLI entry point for the evaluation pipeline.

Run from the corrdiff repo root::

    # Single file (primary path)
    python -m analysis.eval_pipeline.run single --pred FILE.nc --name diffusion

    # Compare two or more files (repeat --pred/--name/--kind, positionally matched)
    python -m analysis.eval_pipeline.run compare \
        --pred A.nc --name diffusion --pred B.nc --name regression --pred C.nc --name static

    # Reproducible run from a YAML config (CLI flags override config values)
    python -m analysis.eval_pipeline.run compare --config analysis/eval_pipeline/configs/diff_vs_reg.yaml

Add ``--wandb`` (with ``--wandb-project/--wandb-group/--wandb-name``) to log metrics,
plots, and the comparison table to wandb in addition to writing them to disk.
"""

from __future__ import annotations

import argparse
import os

from analysis.eval_pipeline import compare as compare_mod
from analysis.eval_pipeline import config as config_mod
from analysis.eval_pipeline import evaluate as evaluate_mod

DEFAULT_RESULTS = os.path.join(os.path.dirname(__file__), "results")


def _add_common(p: argparse.ArgumentParser):
    p.add_argument("--config", default=None, help="Optional YAML config (CLI flags override it).")
    p.add_argument("--channels", nargs="+", default=None,
                   help="Subset of channel names (default: all 4 outputs + wind_speed_10m).")
    p.add_argument("--outdir", default=None, help=f"Output base dir (default: {DEFAULT_RESULTS}).")
    p.add_argument("--n-samples", type=int, default=None, dest="n_samples",
                   help="Number of event/sample timesteps to plot (default: 3).")
    p.add_argument("--event", action="append", default=None, dest="event_times",
                   metavar="YYYY-MM-DD", help="Force a timestep into sample panels (repeatable).")
    p.add_argument("--times-yaml", default=None, dest="times_yaml",
                   help="YAML with {times: [...]} to add as event timestamps (e.g. conf/test_times_2021.yaml).")
    p.add_argument("--wandb", action="store_true", default=False, help="Also log to wandb.")
    p.add_argument("--wandb-project", default=None, dest="wandb_project")
    p.add_argument("--wandb-group", default=None, dest="wandb_group")
    p.add_argument("--wandb-name", default=None, dest="wandb_name")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="analysis.eval_pipeline.run", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("single", help="Evaluate one generated file.")
    s.add_argument("--pred", default=None, help="Path to the generated .nc file.")
    s.add_argument("--name", default=None, help="Display name for the model.")
    s.add_argument("--kind", default=None, choices=["auto", "diffusion", "regression"],
                   help="Model kind (default: auto-detect from ensemble size).")
    _add_common(s)

    c = sub.add_parser("compare", help="Compare two or more generated files.")
    c.add_argument("--pred", action="append", default=None, dest="preds",
                   help="Path to a generated .nc file (repeat for each model).")
    c.add_argument("--name", action="append", default=None, dest="names",
                   help="Display name, positionally matched to --pred (repeatable).")
    c.add_argument("--kind", action="append", default=None, dest="kinds",
                   choices=["auto", "diffusion", "regression"],
                   help="Model kind, positionally matched to --pred (repeatable).")
    _add_common(c)
    return parser


def _require(cfg: dict, keys: list[str]):
    missing = [k for k in keys if not cfg.get(k)]
    if missing:
        raise SystemExit(f"Missing required option(s): {', '.join('--' + m.replace('_', '-') for m in missing)}")


def _make_wandb(cfg: dict, default_name: str):
    if not cfg.get("wandb"):
        return None
    from analysis.eval_pipeline.wandb_logging import WandbLogger
    return WandbLogger(
        project=cfg.get("wandb_project") or "corrdiff-eval",
        group=cfg.get("wandb_group"),
        name=cfg.get("wandb_name") or default_name,
        config={k: v for k, v in cfg.items() if k != "wandb"},
    )


def main(argv=None):
    args = _build_parser().parse_args(argv)
    cli = dict(vars(args))
    command = cli.pop("command")
    config_path = cli.pop("config", None)

    file_cfg = config_mod.load_yaml(config_path) if config_path else {}
    cfg = config_mod.merge(cli, file_cfg)

    # Apply fallback defaults after merge so config can supply them too.
    outdir = cfg.get("outdir") or DEFAULT_RESULTS
    n_samples = cfg.get("n_samples") or 3
    channels = cfg.get("channels")
    event_times = config_mod.resolve_event_times(cfg.get("event_times"), cfg.get("times_yaml"))

    wb = None
    try:
        if command == "single":
            _require(cfg, ["pred", "name"])
            wb = _make_wandb(cfg, cfg["name"])
            evaluate_mod.run_single(
                cfg["pred"], cfg["name"], cfg.get("kind") or "auto", outdir,
                channel_names=channels, n_samples=n_samples, event_times=event_times, wb=wb,
            )
        else:  # compare
            models = config_mod.resolve_models(cfg)
            if len(models) < 2:
                raise SystemExit(
                    "compare needs at least 2 models — pass --pred/--name twice, "
                    "or a models: list / pred_a+pred_b in the config."
                )
            wb = _make_wandb(cfg, "_vs_".join(m["name"] for m in models))
            compare_mod.run_compare(
                models, outdir, channel_names=channels, n_samples=n_samples,
                event_times=event_times, wb=wb,
            )
    finally:
        if wb is not None:
            wb.finish()
    print("\n✓ Done")


if __name__ == "__main__":
    main()
