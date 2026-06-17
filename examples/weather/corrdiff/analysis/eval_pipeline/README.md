

# CorrDiff Evaluation Pipeline

Offline evaluation of CorrDiff generation outputs (`.nc` files). Scores a single model or
compares two or more, using the **core metric/plot engine in `core/`** (the reference CorrDiff
implementations, vendored unchanged) as the source of truth. Writes metrics + plots to disk
and, optionally, logs them to **wandb**.

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r analysis/eval_pipeline/requirements.txt
```
`cartopy` is optional (georeferenced coastlines); everything else is required.

## Usage

Run as a module **from the corrdiff repo root** (so `analysis` is importable):

```bash
# Single file (primary path)
python -m analysis.eval_pipeline.run single \
    --pred generated/all4ens-2021-02-02-and03-1week-3h.nc --name diffusion

# Compare two or more files (repeat --pred/--name/--kind; matched by position)
python -m analysis.eval_pipeline.run compare \
    --pred generated/all4ens-2021-02-02-and03-1week-3h.nc --name diffusion \
    --pred generated/alphav1-reg-comp2021_3h.nc           --name regression \
    --pred generated/static-2019.nc                       --name static-2019

# Reproducible run from a YAML config (CLI flags override config values)
python -m analysis.eval_pipeline.run compare \
    --config analysis/eval_pipeline/configs/diff_vs_reg.yaml
```

### Common flags

| Flag | Default | Description |
|------|---------|-------------|
| `--kind {auto,diffusion,regression}` | `auto` | Model kind. `auto` ⇒ ensemble size > 1 is diffusion, else regression. |
| `--channels NAME ...` | all | Subset of variables (4 outputs + `wind_speed_10m`). |
| `--outdir PATH` | `analysis/eval_pipeline/results` | Output base directory. |
| `--n-samples N` | `3` | Number of timesteps in spatial sample panels. |
| `--event YYYY-MM-DD` | — | Force a timestep into the sample panels (repeatable). |
| `--times-yaml FILE` | — | Add event times from a `{times: [...]}` YAML, e.g. `conf/test_times_2021.yaml`. |
| `--config FILE` | — | YAML run definition; CLI flags override it. |
| `--wandb` + `--wandb-project/-group/-name` | off | Also log metrics, plots, and tables to wandb. |

`--wandb` works fully offline for testing: `WANDB_MODE=offline python -m analysis.eval_pipeline.run single ... --wandb`.

## Regression vs diffusion

Detected from the `ensemble` dimension (size 1 ⇒ deterministic). Deterministic models
report only `rmse/mae/bias/pcc` (CRPS collapses to MAE); ensemble-only quantities
(`spread`, `spread_skill_ratio`, rank histogram) and the spread-skill overlay are skipped /
marked `N/A`.

## Metrics

All metrics come from `analysis.eval_pipeline.core.metrics.MetricsAccumulator`, constructed
with `skip_conditional_metrics=True` because the CWB variables are not precipitation-in-mm —
this also disables the `clamp(min=0)` that would corrupt the signed wind components. Per variable:

| Metric | Notes |
|--------|-------|
| RMSE | global `sqrt(mean(SE))` of the ensemble mean |
| MAE / Bias | mean absolute / signed error |
| PCC | global Pearson correlation |
| CRPS | proper finite-ensemble CRPS (`E\|X−y\| − ½E\|X−X'\|`); == MAE for 1 member |
| Spread / Skill / Spread-Skill ratio | Fortin (2014) `sqrt(mean var)` with `(R+1)/R` correction; ensemble only |
| rank histogram | tie-aware Talagrand counts; ensemble only |

> Note: these differ from the old `analysis/generation/` numbers by definition — e.g. RMSE is
> aggregated globally here vs. averaged per-timestep there. The `core` value is the trusted one.

## Plots

- **Per variable** `diagnostics_<var>.png` — the `plot_diagnostic_panel` 2×3 (spread-skill,
  rank histogram, Q-Q, log-histogram, RAPSD, log-PDF).
- **Event panels** `event_<stamp>.png` — truth / ensemble mean / spread maps per variable.
- **Comparison** (`compare`): `compare_rapsd.png`, `compare_distributions.png`,
  `compare_spread_skill.png`, `compare_metric_bars.png`, `compare_spatial_<stamp>.png`,
  plus `metrics_comparison.csv`. All overlay 2+ models on one figure (the first model is
  the reference for truth, grid, and event-timestep selection); the metric table gets one
  column per model.
- **Comparison extras**: `compare_rapsd_ratio.png` — model PSD ÷ truth PSD on a linear
  axis (perfect = 1.0; shows per-scale power deficits the log-log overlay hides);
  `compare_qq.png` — model vs truth quantiles per variable (perfect = diagonal;
  magnifies tail/extreme differences); `compare_delta_heatmap.png` — RMSE/MAE Δ% per model relative to
  the model named `baseline` (falls back to the reference), red = worse;
  `compare_error_maps_<var>.png` — per-pixel time-mean bias, RMSE, and ΔRMSE-vs-baseline
  maps, one column per model (shows *where* on the grid models differ);
  `compare_monthly_cycle.png` / `compare_diurnal_cycle.png` — domain-mean RMSE and bias
  bucketed by month / hour of day. The cycle plots are controlled by
  `--temporal-cycles {auto,on,off}` (config key `temporal_cycles`); `auto` (default)
  skips them when the run has <50 timesteps or <2 distinct months/hours.

## Output layout

```
results/<stem>/                      # single
    metrics.json   metrics.csv   plots/*.png
results/<stemA>_vs_<stemB>[_vs_...]/  # compare (2+ models; long names are shortened)
    metrics_comparison.csv   plots/*.png
```
