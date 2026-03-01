# Generation Output Analysis

Analysis of CorrDiff model **generation/inference outputs** — comparing the diffusion model
(4 ensembles) vs. the regression-only baseline (1 ensemble) over Taiwan.

## Input files

| File | Model | Ensembles | Timesteps |
|------|-------|-----------|-----------|
| `output/gen_taiwan/all4ens-2021-02-02-and03-1week-3h.nc` | Diffusion | 4 | 84 |
| `output/gen_taiwan/regonly-2021-02-02-and03-1week-3h.nc` | Regression | 1 | 84 |

Each file contains three NetCDF groups:
- **truth** — CWA ground truth (reflectivity, T2m, U-wind, V-wind)
- **prediction** — model output with shape `(ensemble, time, y=448, x=448)`
- **input** — coarse ERA5 conditioning fields

## Scripts

Run from the **repo root**:

```bash
# 1. Compute metrics (saves .nc and .csv to scores/)
python analysis/generation/metrics.py

# 2. Generate all plots (reads from scores/, saves PNGs to plots/)
python analysis/generation/plots.py
```

## Metrics computed (`metrics.py`)

Per variable (`maximum_radar_reflectivity`, `temperature_2m`, `eastward_wind_10m`,
`northward_wind_10m`, `wind_speed_10m`):

| Metric | Description |
|--------|-------------|
| RMSE | Root-mean-square error of ensemble mean vs. truth (spatial avg) |
| MAE | Mean absolute error of ensemble mean vs. truth |
| Bias | Mean signed error: prediction − truth |
| CRPS | Continuous ranked probability score (proper probabilistic metric) |
| Spread | Mean ensemble standard deviation across members |
| Spread/Skill | Spread ÷ RMSE (ideal = 1; >1 overconfident) |
| Pattern Corr | Spatial pattern correlation per timestep |

Output files:
- `scores/diffusion_metrics.nc` — time series of all metrics, diffusion model
- `scores/regression_metrics.nc` — time series of all metrics, regression model
- `scores/summary.csv` — time-mean scalar table (model × variable × metric)

## Plots generated (`plots.py`)

| File | Description |
|------|-------------|
| `rmse_timeseries.png` | RMSE over time, all variables, diffusion vs regression |
| `crps_timeseries.png` | CRPS over time, all variables |
| `bias_maps.png` | Spatial mean bias maps for each model and variable |
| `error_maps.png` | Spatial mean absolute error maps |
| `spread_skill.png` | Ensemble spread vs. RMSE scatter (diffusion model) |
| `power_spectra.png` | Power spectra: KE, T2m, Reflectivity — truth vs. both models |
| `sample_panel_t*.png` | Side-by-side panels: truth / regression / diffusion-ens-mean |
| `summary_bar_rmse.png` | Bar chart: time-mean RMSE per variable |
| `summary_bar_crps.png` | Bar chart: time-mean CRPS per variable |
| `summary_bar_mae.png` | Bar chart: time-mean MAE per variable |
