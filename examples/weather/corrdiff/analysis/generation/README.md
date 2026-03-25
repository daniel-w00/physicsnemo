# Generation Output Analysis

Analysis of CorrDiff model generation outputs. Supports evaluating a single model or comparing any two models side-by-side.

## NetCDF Output Format

Each generation `.nc` file has three groups:
- **truth** — CWA ground truth `(time, y=448, x=448)`
- **prediction** — model output `(ensemble, time, y=448, x=448)`
- **input** — coarse ERA5 conditioning fields (optional)

Variables: `maximum_radar_reflectivity`, `temperature_2m`, `eastward_wind_10m`, `northward_wind_10m`

## Usage

Run from the **repo root**. Model spec format: `name:path[:checkpoint_step]`

```bash
# Single model evaluation
python analysis/generation/metrics.py --model "regression:output/gen_taiwan/regonly.nc:800k"
python analysis/generation/plots.py   --model "regression:output/gen_taiwan/regonly.nc:800k"

# Two-model comparison (any combination: reg vs reg, diff vs diff, reg vs diff)
python analysis/generation/metrics.py \
    --model "diffusion:output/gen_taiwan/all4ens.nc:1400k" \
    --model "regression:output/gen_taiwan/regonly.nc:800k"
python analysis/generation/plots.py \
    --model "diffusion:output/gen_taiwan/all4ens.nc:1400k" \
    --model "regression:output/gen_taiwan/regonly.nc:800k"
```

Optional flags: `--outdir path/to/base` (override output base), `--n-samples N` (sample panels, default 3).

## Output Structure

```
analysis/generation/results/
    regression/                       # single model
        scores/metrics.nc
        plots/*.png
    diffusion_vs_regression/          # two-model comparison
        scores/diffusion_metrics.nc
        scores/regression_metrics.nc
        plots/*.png
```

## Metrics

Per variable, per timestep (spatially averaged):

| Metric | Description |
|--------|-------------|
| RMSE | Root-mean-square error of ensemble mean vs. truth |
| MAE | Mean absolute error |
| Bias | Mean signed error (prediction − truth) |
| CRPS | Continuous ranked probability score (= MAE for single-ensemble) |
| Spread | Mean ensemble standard deviation |
| Spread/Skill | Spread ÷ RMSE (ideal = 1) |
| Pattern Corr | Spatial pattern correlation per timestep |

## Plots

| File | Description |
|------|-------------|
| `rmse_timeseries.png` | RMSE over time, all variables |
| `crps_timeseries.png` | CRPS over time, all variables |
| `bias_maps.png` | Time-averaged spatial bias maps |
| `error_maps.png` | Time-averaged spatial MAE maps |
| `spread_skill.png` | Spread vs. RMSE scatter (ensemble models only, skipped if none) |
| `sample_panel_t*.png` | Truth + model predictions at selected timesteps |
| `summary_bar_{metric}.png` | Time-mean RMSE/MAE/CRPS per variable (grouped bars) |
