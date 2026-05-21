# Project Memory

## Environment

- Apptainer image: `~/apptainer/corrdiff_10_02.sif` (zarr 2.18.3, Python 3.12)
- Run code via: `apptainer exec ~/apptainer/corrdiff_10_02.sif python3 ...`
- CWB dataset (zarr v2): `/data/42-julia-hpc-rz-lsx/sih25nq/downscaling/CorrDiff/cwa_dataset/cwa_dataset.zarr`
  - Shape: cwb (time, 4, 450, 450), era5 (time, 20, 450, 450)
  - ~35064 timesteps, ~30263 valid

## Alex cluster (NHR@FAU)

Login node hostname: `alex1.nhr.fau.de`. Detect via `$CLUSTER == alex` or `hostname` starting with `alex`.

Paths on Alex differ from Julia2 — the data/container locations above are Julia2; on Alex use:

- **Europa training dataset (zarr v3, CWA-shape)**: `/anvme/workspace/b214cb18-ws-daniel2/wuerzburg450_corrdiff.zarr` (345 GB)
  - Same schema as the Julia2 store documented in [europa_dataset_for_corrdiff.md](../europa_dataset_for_corrdiff.md): cwb (time, 4, 450, 450), era5 (time, **12**, 450, 450), 35064 hourly steps 2018–2021.
  - Store is **zarr v3** (has `zarr.json`, no `.zmetadata`) — must be opened with `corrdiff_zarr3.sif`.
  - Use this path on Alex instead of the `/data/42-julia-hpc-rz-lsx/s373395/...` path referenced in `europa_dataset_for_corrdiff.md`.
- **Apptainer images** live in `~/apptainer/`:
  - `corrdiff_10_02.sif` — zarr 2.18.3 image (matches the Julia2 setup).
  - `corrdiff_zarr3.sif` — zarr 3.x image; use when running code that requires zarr v3 (see "zarr v2 / v3 Compatibility" below).
- Workspace root `/anvme/workspace/b214cb18-ws-daniel2/` also has an empty `europa/` placeholder folder — the training store sits one level up, not inside it.

## zarr v2 / v3 Compatibility (`datasets/cwb.py`)

Tested 2026-03-01 with zarr 3.1.5 against the real CWB zarr-v2 store — **all API patterns pass**.

- `zarr.open_consolidated(path)` — works without `zarr_format=2` (zarr v3 auto-detects v2 stores)
- Array indexing, `.size`, `.shape`, `.ndim`, `.attrs` — fully compatible
- String arrays (`cwb_variable`, `era5_variable`): zarr v3 returns numpy scalars instead of plain Python strings, but `np.where(variable == "...")` comparisons still work correctly
- **One fix needed**: `cwb.py:_get_channel_meta` must use `str(variable)` (already applied) — without it, `NetCDFWriter` crashes during generation with a numpy ufunc error (`variable.name + variable.level`). Training is unaffected since it never calls `NetCDFWriter`.
- The `str()` fix is backwards-compatible with zarr v2

> Note: `pip install zarr>=3` inside apptainer also upgrades `numcodecs`, which breaks zarr 2.18 on the next run. Always `pip uninstall zarr numcodecs` afterwards to restore the image state.

## Evaluation timestamps (val / test split)

The default CWB pipeline uses `is_2021` / `is_not_2021` to split — i.e. **all of 2021 = validation**, which means any timestamp later used for "test" generation has already been seen during validation and biases checkpoint selection. To get a clean disjoint split:

- `make_eval_timestamps.py` samples two disjoint random sets of 256 timestamps each from the valid-2021 pool (cwb_valid AND all era5_valid). With `--with-events`, 10 curated typhoon timestamps (In-Fa, Lupit, Chanthu — hardcoded as `TYPHOON_TIMES` constant in the script) are force-included in the test set and appended at the end of the sorted list. Edit `TYPHOON_TIMES` directly to adjust.
- Outputs `conf/val_times_2021.yaml` and `conf/test_times_2021.yaml`, each a flat YAML mapping `{ times: [iso, iso, ...] }`. Regenerate with:
  ```bash
  apptainer exec ~/apptainer/corrdiff_10_02.sif python3 make_eval_timestamps.py \
      --output-val conf/val_times_2021.yaml \
      --output-test conf/test_times_2021.yaml --with-events
  ```

`datasets/cwb.py:get_zarr_dataset` accepts `include_times: list[str]` — when set, the dataset is filtered to exactly those ISO timestamps and the year-2021 split is bypassed (forces `all_times=True` internally).

Hydra wiring pattern (see `conf/config_training_taiwan_regression-alex.yaml`):
```yaml
defaults:
  - val_times_2021@_val_times       # also: test_times_2021@_test_times for generation

validation:
  train: false
  all_times: false
  include_times: ${_val_times.times}
```

To use the test set in generation, add `test_times_2021@_test_times` to the generate config's `defaults:` and set `generation.times: ${_test_times.times}` — no code change needed (`generate.py` already consumes `cfg.generation.times`).

