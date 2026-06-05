# `exp_static2019_1_reg` — Status

**Last updated:** 2026-06-05 10:35 CEST
**State:** ✅ All 12 runs launched (`v1`) and healthy. 8 N1 (bs=4) + 4 N8 (bs=4, compile, resident-emb fix). 24 generate jobs `afterok`, no orphans.

The experiment: 12 **regression** CorrDiff runs, single **static 2019** embedding year, ablating
region (Taiwan/Europa) × embedding (Alpha 64ch / OLMO 128ch) × fusion (concat / emb-branch) × resolution (N1 / N8).
Plan: `~/.claude/plans/i-want-to-do-enchanted-pretzel.md`.

## Launch

```bash
cd ~/corrdiff
jobs/exp_static2019_1_reg/submit_experiment.sh v1 all
```

Per run: 1 train job (4×A100; N8 adds `--constraint=a100_80`) + 2 `afterok` generate jobs
(test split → `test2021.nc`, full-year 1400h → `year2021.nc`). Generate jobs guard on the final
checkpoint existing, so they fail harmlessly if training isn't done.

## Job IDs (v1, submitted 2026-06-05)

| run | train | gtest | gyear | constraint |
|-----|-------|-------|-------|-----------|
| taiwan_alpha_concat_N1 | 3701894 | 3701895 | 3701896 | — |
| taiwan_alpha_branch_N1 | 3701897 | 3701898 | 3701899 | — |
| taiwan_olmo_concat_N1  | 3701900 | 3701901 | 3701902 | — |
| taiwan_olmo_branch_N1  | 3701903 | 3701904 | 3701905 | — |
| europa_alpha_concat_N1 | 3701906 | 3701907 | 3701908 | — |
| europa_alpha_branch_N1 | 3701909 | 3701910 | 3701911 | — |
| europa_olmo_concat_N1  | 3701912 | 3701913 | 3701914 | — |
| europa_olmo_branch_N1  | 3701915 | 3701916 | 3701917 | — |
| taiwan_alpha_branch_N8 | **3702401 (bs=4, compile)** | 3702402 | 3702403 | a100_80 |
| taiwan_olmo_branch_N8  | **3702404** | 3702405 | 3702406 | a100_80 |
| europa_alpha_branch_N8 | **3702407** | 3702408 | 3702409 | a100_80 |
| europa_olmo_branch_N8  | **3702410** | 3702411 | 3702412 | a100_80 |
(earlier N8 IDs 3701918/43/81, 3702054/57/60/63 etc. were cancelled iterations — see "N8 perf journey" below.)

> **Final N8 config: `batch_size_per_gpu=4` + `torch_compile=true`** (compile in the 4 N8 configs;
> bs=4 via `EXTRA_ARGS='++training.hp.batch_size_per_gpu=4'`). grad-accum keeps `total_batch_size=256`
> → final checkpoint still `UNet.0.2000128.mdlus`. ~8.8 s / 256-sample tick, ~34 GB peak → **~19 h** (one chunk).
> N1 runs: base default 8 workers, bs=4, no compile-override (concat configs set compile=true, branch=false per plan).

## N8 perf journey (why it took several resubmits)

The N8 emb-branch runs were initially **non-viable** and needed two real fixes + tuning:

1. **OOM at bs=4** (old code): the 3.3 GB static embedding was held *per batch* too → resubmitted bs=2.
2. **GPU starvation (~1 step/min, ≈170 days to 2M)** — ROOT CAUSE: `datasets/cwb.py:__getitem__`
   returned the full **(64, 3584, 3584) ≈ 3.3 GB static embedding for EVERY sample** through the
   DataLoader (worker IPC + host→GPU per step), for a tensor identical across all samples.
   **FIX (code):** deliver it once, **GPU-resident, broadcast over the batch** — new `static_embedding`
   property in `cwb.py`; `train.py`/`generate.py` load it once and `expand(B, …)` per step.
   → **~130× speedup** (1 step/min → 17.5 samples/s). N8 mem dropped 45→40 GB, CPU mem 146→9 GB.
3. **`torch_compile=true`** on N8: a further **~35% + half memory** (14.6→9.4 s/tick, 40→19 GB at bs=2).
   Graph-breaks on the custom `SongUNetEmbBranch` fall back to eager; numerically safe; checkpoints compatible.
4. **bs=2 vs bs=4** (with compile): N8 is compute-bound, so bs=4 is only ~6% faster (9.4→8.8 s/tick)
   but +15 GB (19→34 GB). Chose **bs=4** since `a100_80` has ample headroom and it's the fastest.
5. **TF32 red herring:** the persistent `_inductor` "TF32 not enabled" warning is an **inductor
   compile-worker artifact** — the real training process has `cuda.matmul.allow_tf32=True` already
   (container default `'high'`). And it's irrelevant to speed anyway: these are **conv-dominated** nets,
   convs use the separate `cudnn.allow_tf32` (also True). Enabling/disabling matmul TF32 changed N8 time 0%.
   No code change kept (reverted the no-op `set_float32_matmul_precision`).

## Known events

- **2026-06-05:** `taiwan_alpha_branch_N8` (job 3701918) **OOM'd at bs=4** on 80 GB A100 (76.9 GB used,
  died on first step's 784 MiB alloc). Cancelled its orphaned generate jobs (3701919/3701920), wiped the
  empty `v1/`, and **resubmitted at bs=2** (job 3701943 + generates 3701944/3701945) — runs clean, no OOM.
  ⚠️ The **other three N8 runs may OOM the same way** — if so, apply the identical bs=2 resubmit
  (`taiwan_olmo_branch_N8` 3701921, `europa_alpha_branch_N8` 3701924, `europa_olmo_branch_N8` 3701927).
  Resubmit recipe (single run, bs=2, with its 2 afterok generates) is below under "Resume / recovery".

## Output paths (`$WORK = /home/atuin/b214cb/b214cb18`)

- checkpoints: `$WORK/checkpoints/exp_static2019_1_reg/<run>/v1/checkpoints_regression/UNet.0.2000128.mdlus`
- generated:   `$WORK/generated/exp_static2019_1_reg/<run>/v1/{test2021.nc, year2021.nc}`
- final checkpoint name is deterministic (`total_batch_size=256`, `training_duration=2_000_000` ⇒ step 2_000_128).

## Files created

- **Configs (conf root, flattened):** `exp_static2019_1_reg_common_train.yaml`, `exp_static2019_1_reg_common_gen.yaml`,
  `exp_static2019_1_reg_train_<run>.yaml` ×12, `exp_static2019_1_reg_gen_<run>.yaml` ×12,
  `exp_static2019_1_reg_year2021_random1400.yaml` (committed 1400-hour set, seed 0, disjoint from val/test).
- **Jobs:** `jobs/exp_static2019_1_reg/{train.slurm, generate.slurm, submit_experiment.sh}`.
- **Scripts:** `make_year_sample.py` (new); one-line read-only fix in `make_eval_timestamps.py`
  (`zarr.open_consolidated(..., mode="r")`).

## Key decisions / gotchas (learned during bring-up)

- **Configs live at conf ROOT, not a subdir.** A subdir forced leading-slash group defaults, which broke
  `train.py:133` `runtime.choices.dataset` and the `training: ${model}` interpolation. Flattened names fix both.
- **`++` force-add** is required for the injected overrides (`training.io.checkpoint_dir`,
  `generation.io.{reg_ckpt_filename,output_filename}`) — those keys aren't in the base struct, so plain `=` fails.
- **Per-run knobs must NOT be in `_common_train`.** It's `# @package _global_`, which Hydra 1.3 merges AFTER
  `_self_`, so per-run configs can't override it. `torch_compile` (concat=true/branch=false) and
  `dataloader_workers` (N8=0) are therefore set in each run config.
- **concat → no `model.model_args`** (plain UNet); **branch → `SongUNetEmbBranch`** with
  `alpha_earth_channels` = 64 (alpha) / 128 (olmo).
- Smoke test on a real A100 confirmed: composition, overrides, `runtime.choices`, data+embedding load, and the
  `EMBEDDINGS: ON (single file, static for all samples)` banner with the correct path.

## Monitoring

```bash
squeue -u $USER -o "%.10i %.30j %.10T %.16R"
tail -f output/alex/jobout/job-<id>.out
ls $WORK/checkpoints/exp_static2019_1_reg/*/v1/checkpoints_regression/
```

## Resume / recovery

- **Hit 24h wall:** `submit_experiment.sh v1 <run>` → training auto-resumes from `v1/`; premature generate
  jobs fail the guard harmlessly. When `UNet.0.2000128.mdlus` exists → `submit_experiment.sh v1 <run> --gen-only`.
- **N8 OOM:** resubmit that run's training with `EXTRA_ARGS='++training.hp.batch_size_per_gpu=2'` (or `=1`);
  grad-accum keeps `total_batch_size=256`, so the checkpoint name is unchanged.
- **Fresh restart from scratch:** new VERSION (e.g. `v2`) — all paths/tags pick it up.

## Year-sample regeneration (if ever needed)

```bash
apptainer exec --bind /anvme:/anvme ~/apptainer/corrdiff_zarr3.sif python3 make_year_sample.py \
  --data-path /anvme/workspace/b214cb18-ws-daniel3/b214cb18-ws-daniel-cwb-1779073861/cwa_dataset/cwa_dataset.zarr \
  --exclude conf/val_times_2021.yaml --exclude conf/test_times_2021.yaml \
  --output conf/exp_static2019_1_reg_year2021_random1400.yaml --n 1400 --seed 0
```
(Note: data-path must end in `/cwa_dataset.zarr`, and `/anvme` must be bound into the container.)
