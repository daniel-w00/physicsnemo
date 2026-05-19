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
