# Generation Output Analysis

Analysis of CorrDiff model generation outputs. Supports evaluating a single model or comparing multiple models side-by-side.

## Quick Start

```bash
# Single command: compute all metrics and generate all plots
python analysis/generation/evaluate.py --model "regression:generated/alphav1-reg-comp2021_3h.nc"
```

Results are written to `analysis/generation/results/alphav1-reg-comp2021_3h/`.

## Usage

Run from the **repo root**. Model spec format: `name:path[:checkpoint_step]`

### Single model

```bash
python analysis/generation/evaluate.py \
    --model "regression:generated/alphav1-reg-comp2021_3h.nc"
```

### Multi-model comparison

```bash
python analysis/generation/evaluate.py \
    --model "diffusion:generated/all4ens-2021-02-02-and03-1week-3h.nc" \
    --model "regression:generated/alphav1-reg-comp2021_3h.nc"
```

### With event markers in sample panels

```bash
python analysis/generation/evaluate.py \
    --model "diffusion:generated/all4ens-2021-02-02-and03-1week-3h.nc" \
    --event "2021-02-02:Event label here"
```

### Optional flags

| Flag | Default | Description |
|------|---------|-------------|
| `--outdir PATH` | `analysis/generation/results/` | Override output base directory |
| `--n-samples N` | `3` | Number of timesteps in sample panels |
| `--event DATE:LABEL` | — | Force a timestep into sample panels (repeatable) |

### Run steps separately

`metrics.py` and `plots.py` can also be called individually if you only need one step:

```bash
python analysis/generation/metrics.py --model "regression:generated/alphav1-reg-comp2021_3h.nc"
python analysis/generation/plots.py   --model "regression:generated/alphav1-reg-comp2021_3h.nc"
```

## NetCDF Output Format

Each generation `.nc` file has three groups:
- **truth** — CWA ground truth `(time, y=448, x=448)`
- **prediction** — model output `(ensemble, time, y=448, x=448)`
- **input** — coarse ERA5 conditioning fields (optional)

Variables: `maximum_radar_reflectivity`, `temperature_2m`, `eastward_wind_10m`, `northward_wind_10m`

## Output Structure

```
analysis/generation/results/
    regonly/                          # single model (named after .nc file stem)
        scores/metrics.nc             # scalar time-series + spatial maps
        plots/*.png
    all4ens_vs_regonly/               # multi-model comparison
        scores/diffusion_metrics.nc
        scores/regression_metrics.nc
        plots/*.png
```

The `metrics.nc` contains both the per-timestep scalar scores and the time-averaged 2D spatial maps (bias, MAE, lat/lon grids).

## Metrics

Per variable, per timestep (spatially averaged):

| Metric | Description |
|--------|-------------|
| RMSE | Root-mean-square error of ensemble mean vs. truth |
| MAE | Mean absolute error of ensemble mean |
| Bias | Mean signed error (prediction − truth) |
| CRPS | Continuous ranked probability score (= MAE for single-member) |
| Spread | Mean ensemble standard deviation |
| Spread/Skill | Spread ÷ RMSE (ideal = 1.0) |
| Pattern Corr | Spatial pattern correlation per timestep |

## Plots

| File | Description |
|------|-------------|
| `rmse_timeseries.png` | RMSE over time, all variables |
| `crps_timeseries.png` | CRPS over time, all variables |
| `bias_maps.png` | Time-averaged spatial bias maps |
| `error_maps.png` | Time-averaged spatial MAE maps |
| `spread_skill.png` | Spread vs. RMSE scatter (ensemble models only) |
| `sample_panel_*.png` | Truth + predictions at selected timesteps |
| `summary_bar_{metric}.png` | Time-mean RMSE/MAE/CRPS per variable (grouped bars) |
| `rank_histogram_{name}.png` | Talagrand diagram for ensemble calibration (ensemble only) |
