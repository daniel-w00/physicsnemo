# `exp_static2019_1_reg` — Status

**Last updated:** 2026-06-07 CEST
**State:** ✅✅ **COMPLETE.** All 12 runs trained to the final step `UNet.0.2000128.mdlus` and all 24 generation
outputs (`test2021.nc` + `year2021.nc`, all non-empty) are written under `$WORK/generated/exp_static2019_1_reg/<run>/v1/`.
Queue empty, no jobs pending. Getting here required fixing the N8 generate bug + recovering 5 train interruptions
(1 transient zstd crash, 4 branch-N1 24h-wall timeouts) — see "Known events" and the wall-time note below.

> **⚠️ For the next experiment: emb-branch N1 runs do NOT fit a 24h wall at 2M steps.** With `torch_compile=false`
> (the branch-N1 ablation setting) they run ~12.0 s/tick vs ~8.8 s/tick for compile-on concat-N1 → ~36% slower,
> needing **~26.5 h** for 2M vs concat's ~19.7 h. All 4 branch-N1 runs timed out ~0.2–0.4M short and had to be
> resumed (auto-resume from last checkpoint + `EXTRA_ARGS='wandb.resume_id=<id>'`). Next time either enable
> `torch_compile=true` for branch-N1 (checkpoints are compatible; the N8 branch runs used it and finished ~20 h)
> or split into 2 walls up front. wandb resume from a checkpoint behind the last logged step just drops the
> re-done overlap segment (monotonic-step warnings) — harmless.

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

- **2026-06-06: branch N1 OOMs on 40 GB cards.** The emb-branch N1 runs (compile OFF per plan) peak
  **~55 GB**, so they fail with `CUDA out of memory` if SLURM places them on a 40 GB A100. Two
  unconstrained branch-N1 jobs landed on 40 GB and died in ~80 s; resubmitted with `--constraint=a100_80`.
  **All 4 `*_branch_N1` runs now require a100_80** (baked into `submit_experiment.sh`'s TR_CONSTRAINT).
  concat N1 is fine on 40 GB (compile ON, no branch, ~24 GB). Did NOT switch branch N1 to compile (which
  would fit 40 GB + be faster) because 2 of the 4 were already 31% done at compile-off — matching them
  keeps the branch-N1 ablation cell consistent and avoids discarding ~8 h × 2 of compute.
- **Slow-queue trick:** unconstrained jobs are weight-steered to the busier 40 GB pool (node Weight 70 vs
  80 GB's 90). Adding `--constraint=a100_80` (or in-place `scontrol update JobId=<id> Features=a100_80`)
  moves a job to the less-contended 80 GB pool — observed pulling projected starts ~16 h earlier and
  starting an instantly-eligible job immediately. In-place update keeps job IDs + afterok deps intact.
- **Storage outage 2026-06-05 ~17:08** killed the 7 then-pending N1 jobs (resubmitted); running jobs unaffected.
- **2026-06-06: transient `Zstd decompression error` killed `europa_alpha_concat_N1` (job 3703970) at step ~1,000,192 after 11h15.**
  A DataLoader worker raised `RuntimeError: Zstd decompression error: invalid input data` while reading
  `cwb_variable` (a tiny static coord array) in `get_target_normalizations_v3_europa` (`cwb.py:185`); the
  uncaught worker exception crashes the whole 4-GPU DDP job. **Root cause = transient networked-store I/O
  glitch, NOT corrupt data:** the store re-reads cleanly now (verified all coord arrays decode), and all 12
  runs share the same store — only this one tripped, which is the signature of a rare per-read dice roll
  (each run does ~10⁴–10⁵ chunk reads/run; likely related to the same flaky-storage window as the 17:08 outage).
  **Decision: no code change.** A dataloader retry-wrapper was prototyped + reverted — the failure is rare and
  transient, not worth defensive code in the shared hot path. **Recovery = plain resubmit** (auto-resumes from
  the step-1M checkpoint in `v1/`), adding `EXTRA_ARGS='wandb.resume_id=etjed5h9'` so it continues the existing
  wandb run instead of forking a new one. If the glitch recurs on other long runs, just resubmit them the same way.
- **2026-06-06: all 4 N8 generation runs (8 jobs) failed — real code bug, now fixed.** Every `*_branch_N8`
  gtest/gyear died in <5 min with `ValueError: Model type 'SongUNetEmbBranch' is not supported` from
  `Module.from_checkpoint`. `generate.py` never ran the model-registration block that `train.py` has
  (`_diffusion_module.SongUNetEmbBranch = …` + `UNet._wrapped_classes |= {...}`). **FIX:** added that block to
  `generate.py` (after the `from physicsnemo import Module` import). Affects all 8 *branch* runs (N8 + branch-N1
  generations); concat runs were unaffected (plain `UNet`). N8 gen jobs resubmitted gen-only: 3706164–3706171.


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
