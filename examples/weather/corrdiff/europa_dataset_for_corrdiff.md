# Europa dataset for CorrDiff training

Quick-orienting summary for Claude Code working on the CorrDiff training side. This dataset is the European (Würzburg) counterpart to NVIDIA's CWA dataset and is designed to be loaded by the same CorrDiff training pipeline with only minimal, well-defined config changes.

For the full thesis-grade reasoning behind every design choice, see
[`design_choices.md`](design_choices.md) (notably §§ 1, 3, 7).

## Paths

| Purpose | Path |
|---|---|
| **Training store (CWA-shape)** | `/data/42-julia-hpc-rz-lsx/s373395/europa1/europa-together/wuerzburg450_corrdiff.zarr` |
| Analysis store (per-variable, friendlier for inspection) | `/data/42-julia-hpc-rz-lsx/s373395/europa1/europa-together/wuerzburg450_hourly.zarr` |
| CWA reference (for direct comparison) | `/data/42-julia-hpc-rz-lsx/sih25nq/downscaling/CorrDiff/cwa_dataset/cwa_dataset.zarr` |

Use the **corrdiff store** for training. The analysis store is for sanity checks and plotting; the two are derived from the same underlying data.

## Schema vs CWA — what is the same, what differs

| | Europa (this dataset) | CWA |
|---|---|---|
| Time coverage | 2018-01-01 00:00 .. 2021-12-31 23:00, hourly, 35 064 steps | identical |
| High-res grid | 450 × 450, WRF Mercator, 3 km, centred on Würzburg | 450 × 450, WRF Mercator, 3 km, centred on Taiwan |
| `cwb` shape | `(time, 4, 450, 450) float32`     | identical shape |
| `era5` shape | `(time, **12**, 450, 450) float32` | `(time, **20**, 450, 450) float32` |
| Staggered coords (`XLAT_U/V`, `XLONG_U/V`) | present | present |
| `cwb_center / cwb_scale / era5_center / era5_scale` | present (dask mean / std over all 4 years) | present |
| `cwb_valid / era5_valid` | present, all-True (source store verified 0 NaN) | present, may contain False |

The grid dimensions, dtypes, chunking, coordinate naming, staggered-grid layout, normalization stats, and validity-mask shape all match CWA exactly. The CorrDiff training code therefore reads our store with the same logic; only the channel set differs (next section).

## Channel layout — three deliberate divergences from CWA

The full channel-index table lives in [`design_choices.md` § 7.3](design_choices.md). The high-level differences a CorrDiff dataloader needs to know about:

1. **`cwb` channel 3 is `precipitation_amount_1hr`, not `maximum_radar_reflectivity`.**
   We have no European radar product equivalent to Taiwan's, so the 4th high-res channel is 1-hour accumulated precipitation from WRF (`PREC_ACC_NC`, mm). The label in the zarr is honest (`precipitation_amount_1hr`); update the dataloader's channel-name list accordingly. Pre-trained CWA weights cannot transfer for this channel anyway (radar dBZ vs precipitation mm — fundamentally different distribution).

2. **`era5` has 12 channels, not 20.**
   We only downloaded 500 and 850 hPa from CDS, not 700 and 925. The order still follows CWA (`tcwv`, then `(z, t, u, v)` per pressure level high-to-low, then surface `t2m/u10/v10`). Any code that hard-codes CWA's 20-channel indices must be reworked; use `era5_pressure` and `era5_variable` to drive channel selection.

3. **`era5_z` channels are labeled `geopotential_height` but contain raw ERA5 `z` in m²/s², not metres.**
   This matches CWA's own (slightly loose) convention — verified by reading CWA's `era5_center` values. No conversion is required when loading; if you need geopotential height in metres, divide by g = 9.80665.

## Channel index quick reference

```
cwb  (4 channels):
  0  temperature_2m            (K)          source: WRF T2
  1  eastward_wind_10m         (m/s)        source: WRF U10   (grid-relative)
  2  northward_wind_10m        (m/s)        source: WRF V10   (grid-relative)
  3  precipitation_amount_1hr  (mm)         source: WRF PREC_ACC_NC

era5 (12 channels):
  0  tcwv                  (kg/m²)   surface
  1  geopotential_height   (m²/s²)   500 hPa
  2  temperature           (K)       500 hPa
  3  eastward_wind         (m/s)     500 hPa
  4  northward_wind        (m/s)     500 hPa
  5  geopotential_height   (m²/s²)   850 hPa
  6  temperature           (K)       850 hPa
  7  eastward_wind         (m/s)     850 hPa
  8  northward_wind        (m/s)     850 hPa
  9  temperature_2m        (K)       surface
 10  eastward_wind_10m     (m/s)     surface
 11  northward_wind_10m    (m/s)     surface
```

## Normalization stats (baked into the store)

```
cwb_center  ≈ [283.24, 0.76, 0.31, 0.108]
cwb_scale   ≈ [  8.24, 3.43, 3.26, 0.667]

era5_center ≈ [16.23, 55101, 253.80, 7.82, -0.92, 14481, 277.99, 3.28, 0.25, 283.69, 0.69, 0.36]
era5_scale  ≈ [ 8.10,  1542,   6.38,10.73, 11.11,   795,   6.92, 7.19, 6.02,   8.00, 3.11, 2.92]
```

These were computed over the full 4-year record from the analysis store (`pipeline/build_corrdiff_store.py::_compute_stats`). The CorrDiff loader should normalise inputs as `(x - center) / scale` per channel; both arrays are stored as data variables in the zarr so the loader can read them directly.

## Wind frame caveat

WRF `U10` / `V10` (cwb channels 1, 2) are **grid-relative** Mercator components, while ERA5 `u10` / `v10` (era5 channels 10, 11) are **earth-relative** (eastward / northward). Over the Würzburg domain at ≈50 °N the Mercator rotation angle is small (≤ 1°) and CorrDiff treats both sides as inputs to a learned mapping, so no rotation is needed for training. If wind directions ever need to be compared in a common frame for evaluation, rotate the WRF winds to earth-relative using `XLAT`/`XLONG` (standard WRF post-processing).

## Suggested dataloader changes (minimal patch)

A dataloader written for CWA needs three small edits to consume the Europa store:

1. Replace `"maximum_radar_reflectivity"` with `"precipitation_amount_1hr"` in the cwb channel-name list (or read `cwb_variable` from the zarr directly).
2. Allow 12 era5 channels in the input (drop hard-coded `20` or `range(20)`; read `n_era5_channels` from the store's top-level attribute, or use `len(era5_variable)`).
3. If your dataloader hard-coded pressure levels `[500, 700, 850, 925]`, change to read `era5_pressure` from the store.

No other code changes should be required: dtype, chunking, normalization arrays, validity masks, and grid coords all match CWA's contract.

## Verification status

- All 35 064 hourly steps present, monotonic, no gaps.
- 0 NaN across all data variables in the analysis store (checked by `pipeline/verify_store.py`); validity masks set to all-True on that basis.
- Per-channel means / stds within textbook ranges for mid-latitude Europe (annual T2 ≈ 10 °C, 500 hPa T ≈ −19 °C, 500 hPa Φ/g ≈ 5617 m, etc.).
- Total store size: 343 GB.

## Training setup on Alex (regression-only)

Verified 2026-05-20: training the regression UNet on the Europa store
needs **zero Python code changes** — `datasets/cwb.py::_ZarrDataset` already
reads channel names, pressure levels, and per-channel normalization
stats from the zarr at load time, and `train.py` derives model in/out
channel counts from `dataset.input_channels()` / `output_channels()`.
The Europa-specific divergences (12-not-20 era5 channels, precipitation
at cwb index 3) are absorbed entirely through three config-level knobs.

### Files

| File | Purpose |
|---|---|
| [`conf/base/dataset/europa.yaml`](conf/base/dataset/europa.yaml) | Europa dataset config. `type: cwb` (reuses the CWA loader), `in_channels: [0..11]`, `out_channels: [0,1,2,3]`, earth embeddings off (`embedding_source: none`, `embedding_region: europa`; see [alpha-integrate.md](alpha-integrate.md)). |
| [`conf/config_training_europa_regression-alex.yaml`](conf/config_training_europa_regression-alex.yaml) | Top-level Hydra config. Pulls `dataset: europa`, `model: regression`, hooks `val_times_2021.yaml` into the validation split. `checkpoint_dir: /checkpoints/europa/reg_eu_pure_v1`. |
| [`jobs/alex/europa_regression.slurm`](jobs/alex/europa_regression.slurm) | Alex SLURM script. A100-80 GB, binds `/anvme/workspace/b214cb18-ws-daniel2` 1:1, uses the zarr-v3 container. |

### Channel-config rationale

CWA used `in_channels: [0,1,2,3,4,9,10,11,12,17,18,19]` to pick 12 of its
20 era5 channels. The Europa store already contains *only* those 12
channels in the same physical order (`tcwv`, `(z,t,u,v)` @ 500, `(z,t,u,v)`
@ 850, `t2m`, `u10`, `v10`), so we use `in_channels: [0..11]` straight.
`out_channels: [0,1,2,3]` is unchanged; channel 3 went from
`maximum_radar_reflectivity` (dBZ, ≈[15, 15]) to `precipitation_amount_1hr`
(mm, center≈0.108, scale≈0.667) — same index, different distribution, so
**pre-trained CWA regression weights cannot transfer for the precip
channel.** Train from scratch.

### Normalization

The Europa config uses **`normalization: europa`**, a Europe-tuned
linear rescale defined in `get_target_normalizations_europa`
([cwb.py:57-85](datasets/cwb.py#L57-L85)). It does the following:

| Channel | center | scale | Rationale |
|---|---:|---:|---|
| `temperature_2m`           | 283.24 (empirical) | 8.24 (empirical) | Gaussian-ish; z-score from store stats. |
| `eastward_wind_10m`        | **0** | 3.43 (empirical) | Anchor at natural zero (winds are sign-symmetric); empirical scale unchanged. |
| `northward_wind_10m`       | **0** | 3.26 (empirical) | Same. |
| `precipitation_amount_1hr` | **0** | **5 mm** | Anchor at natural zero; scale=5 chosen so 1/σ² ≈ 0.04, putting precip ~2.7× T2's implicit MSE weight (vs. ~152× under v1). |

Why not the other variants:

- **`v1`** (read empirical center/scale from the zarr) — gives precipitation
  σ ≈ 0.667 mm, which produces an implicit MSE loss weight ~152× that of
  T2 because mm-precip is zero-inflated. The optimizer would burn most
  of its gradient budget on precip pixels.
- **`v2`** (CWA defaults) — its radar branch matches
  `"maximum_radar_reflectivity"`, which doesn't exist on Europa, so
  precip silently falls back to v1 stats (no fix). Its wind branch
  forces `scale=20 m/s`, tuned for Taiwanese typhoons, which actively
  *underweights* European winds (Würzburg gusts rarely exceed 15 m/s).
  Strictly worse than v1 on Europa.

The `europa` variant fixes the cross-channel imbalance with a single
linear rescale and no non-linear transform. It does *not* address the
heavy-tail training-stability concern from individual convective pixels
— that would require a log1p / asinh transform, which is the next-step
refinement analysed in [normalization_design.md](normalization_design.md).

### Validation split — reusing the CWA 256-timestamp list

[`conf/val_times_2021.yaml`](conf/val_times_2021.yaml) was originally
sampled from CWA's valid-2021 pool (6109 timestamps). Every entry is a
2021 hour, and Europa has *all* 2021 hours valid (`era5_valid` and
`cwb_valid` both all-True), so the list is reused verbatim — no
regeneration needed. Hooked in via Hydra:

```yaml
defaults:
  - val_times_2021@_val_times

validation:
  train: false
  all_times: false
  include_times: ${_val_times.times}
```

`get_zarr_dataset` detects `include_times` and bypasses the `is_2021` /
`is_not_2021` filter (forces `all_times=True` internally), so the
training split is the natural complement (2018–2020 + the 2021 hours
**not** in the val list).

### Container

The Europa store is zarr v3 (only `zarr.json`, no `.zgroup` /
`.zmetadata`). The default `~/apptainer/corrdiff_10_02.sif` (zarr 2.18)
**cannot** read it. The SLURM script uses
`~/apptainer/corrdiff_zarr3.sif` (zarr 3.1.5). Spot-checked: opens
cleanly via `zarr.open_consolidated()`, all expected variables present,
shapes/dtypes/stats match this document, `time` units `"hours since
2018-01-01"` decode correctly through `cftime.num2date`.

### Launch

```bash
sbatch jobs/alex/europa_regression.slurm
```

Checkpoints land in `$WORK/checkpoints/europa/reg_eu_pure_v1/` on the
host (mapped to `/checkpoints/europa/reg_eu_pure_v1` inside the
container).
