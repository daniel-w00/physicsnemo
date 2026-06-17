# `exp_static2019_1_diff` — Status

**Last updated:** 2026-06-17 CEST
**State:** 🟢 **Europa complete (6/6 trained + generated). Taiwan 1/6 embedding (concat) + baseline trained.**
Overall **7/12 embedding runs trained, 7/12 generated**, plus the **no-embedding Taiwan baseline diffusion
trained** (2M, not yet generated). Queue empty. Remaining: 5 Taiwan embedding runs + generates.

The experiment: the **diffusion** counterpart of [`exp_static2019_1_reg`](exp_static2019_1_reg_STATUS.md).
Same 12-run matrix — region (Taiwan/Europa) × embedding (Alpha 64ch / OLMO 128ch) × fusion (concat /
emb-branch) × resolution (N1 / N8) — each run trains the **diffusion residual** on top of the matching
embedding-trained regression checkpoint from `exp_static2019_1_reg`. See
[diffusion_with_embeddings.md](diffusion_with_embeddings.md).

> **Key fact:** each diffusion run's `regression_checkpoint_path` points at the **matching** reg run's final
> checkpoint `…/exp_static2019_1_reg/<run>/v1/checkpoints_regression/UNet.0.2000128.mdlus` (channel counts
> match because every reg run was trained with the same embeddings).

## Files

- **Train configs** (12): `conf/exp_static2019_1_diff_train_<run>.yaml`
- **Generate configs** (now **per split**): `conf/exp_static2019_1_diff_gen_<run>_<split>.yaml` for
  `split ∈ {test2021, top64}` (full reg+diffusion, 16 ensembles, `save_input: false`; **N8 use
  `seed_batch_size: 2`**, others 16). Output: `…/<run>/<run>_<split>.nc`.
- **Generator** (idempotent): `jobs/exp_static2019_1_reg/gen_diffusion_configs.py` — `RUNS` matrix (12) ×
  `SPLITS` (test2021, top64). `taiwan_baseline` is NOT in this matrix (one-off config).
- **Train slurm** (generic, 8×A100): `jobs/exp_static2019_1_reg/train_diffusion.slurm`
  — env: `REGION` (cwb|europa), `CONFIG_NAME`, `CKPT_DIR`, `NUM_GPUS` (default 8).
- **Generate slurm**: `jobs/alex_eu_generate.slurm` — now **region-aware** (`REGION` env, default europa),
  `CONFIG_NAME`-driven, binds `$HPCVAULT/regrid2` for embeddings, and `NUM_GPUS` drives `--nproc_per_node`
  (default 1; generate.py shards ensemble seeds across ranks and writes from rank 0). For >1 GPU also pass
  `--gres=gpu:a100:N` on the CLI (the `#SBATCH` gres can't read env).

## Progress

| run | train | generate (test2021.nc) |
|-----|-------|------------------------|
| europa_alpha_concat_N1 | ✅ | ✅ **56 GB** (only one with `save_input: true`) |
| europa_olmo_concat_N1  | ✅ | ✅ 27 GB |
| europa_alpha_branch_N1 | ✅ | ✅ 27 GB |
| europa_olmo_branch_N1  | ✅ | ✅ 27 GB |
| europa_alpha_branch_N8 | ✅ | ✅ 27 GB |
| europa_olmo_branch_N8  | ✅ | ✅ 27 GB |
| taiwan_alpha_concat_N1 | ✅ | ✅ 27 GB |
| taiwan_alpha_branch_N1 | ⬜ | ⬜ |
| taiwan_olmo_concat_N1  | ⬜ | ⬜ |
| taiwan_olmo_branch_N1  | ⬜ | ⬜ |
| taiwan_alpha_branch_N8 | ⬜ | ⬜ (a100_80) |
| taiwan_olmo_branch_N8  | ⬜ | ⬜ (a100_80) |
| **taiwan_baseline** (no-embedding) | ✅ (3747099, 13h45m) | ⬜ no gen config/job yet |

Outputs at `/home/vault/b214cb/b214cb18/generated/exp_static2019_1_diff/<run>/<run>_<split>.nc`
(older Europa/Taiwan-concat outputs predate the rename and sit at `…/<run>/test2021.nc`).

**Baseline:** `taiwan_baseline` is the no-embedding diffusion reference (config
`conf/exp_static2019_1_diff_train_taiwan_baseline.yaml`, conditioned on the no-embedding
`exp_static2019_1_reg/taiwan_baseline/.../UNet.0.2000128.mdlus`). 2M steps, `hr_mean_conditioning=true` —
so it's directly comparable to the Taiwan embedding diffusion runs (unlike the older `taiwan/pure_*` 1.5M /
`hrm_false` checkpoints). One-off, not in the 12-run generator matrix. Still needs a generate config + job.

## Measurements / decisions

- **batch_size_per_gpu: 4** for all. bs8 = identical throughput (~41 samples/s, compute-bound) but 2× memory;
  `common_train` pins bs4. Diffusion concat trained ~13.5 h for 2M; branch ran fine too.
- **torch_compile: true for ALL runs** (changed from the reg setting). Branch graph-breaks → eager (safe);
  compile keeps everything inside the 24 h wall. No branch-N1 wall overrun occurred.
- **save_input: false** everywhere except the first europa_alpha_concat run (56 GB). Regenerate it if you want
  the uniform 27 GB footprint; otherwise fine.
- **40 GB is enough for alpha_branch_N1 training** (measured peak 37.8 GB reserved, 1-GPU probe = per-GPU DDP
  footprint). OLMO branch (128ch) and N8 still want a100_80.
- **N8 generation needs a100_80 + small seed_batch_size + multiple GPUs.** Single-GPU N8 gen at
  `seed_batch_size: 2` peaked **42.6 GB** (>40 GB) and was too slow (8 passes/timestamp → 6 h timeout).
  Fix: run on **8 GPUs** (`NUM_GPUS=8`, `--gres=gpu:a100:8`) so the 16 ensembles run as one parallel pass
  (2/GPU) → ~8× faster, fits, finishes well under 6 h.

## Known events

- **2026-06-12: all 4 branch diffusion runs failed at startup** (`ValueError: Model type 'SongUNetEmbBranch'
  is not supported`). `train.py`/`generate.py` registered the class only into `UNet._wrapped_classes`, not the
  diffusion wrapper `EDMPrecondSuperResolution._wrapped_classes`. **Fixed** (one line each) + verified. See
  the [[diffusion-branch-wrapped-classes-fix]] memory.
- **2026-06-13: alpha_branch_N1 train died in 39 s** with `init_process_group device_id…` — it landed on a
  **bad `maint` node (a0701)**. Resubmitted with `--exclude=a0701`; ran clean → 27 GB output.
- **2026-06-13: N8 generates OOM'd** (`seed_batch_size: 16` → tried 49 GiB on 80 GB). Dropped N8 to
  `seed_batch_size: 2`.
- **2026-06-15: N8 generates TIMEOUT** at 6 h single-GPU (got 176/266 & 119/266). Switched to 8-GPU; both
  completed after the cluster maintenance drain cleared → 27 GB each.

## ⚠️ Watch-outs / remaining work

- **Taiwan runs (5 left): not started.** Launch as below with `REGION=cwb`. Generation now works for cwb
  (the generate slurm is region-aware). N8 gen needs the 8-GPU recipe.
- **N8 gen recipe**: `--gres=gpu:a100:8 NUM_GPUS=8`, config already has `seed_batch_size: 2`.
- a100_80 / a100_40 nodes periodically drain for maintenance (`drng@`) → long `ReqNodeNotAvailable` pends;
  jobs start automatically when nodes return.

## Launch (per run, manual)

Train:
```bash
cd ~/corrdiff
run=taiwan_olmo_concat_N1            # REGION: taiwan→cwb, europa→europa
sbatch -J ${run}_diff_tr \
  --export=ALL,REGION=cwb,CONFIG_NAME=exp_static2019_1_diff_train_${run},CKPT_DIR=/checkpoints/exp_static2019_1_diff/${run}/v1,NUM_GPUS=8 \
  jobs/exp_static2019_1_reg/train_diffusion.slurm
```
Branch / N8 add `--constraint=a100_80` (alpha_branch_N1 fits 40 GB → omit it).

Generate (after the matching train finishes):
```bash
# N1 / concat (1 GPU):
sbatch -J gen_${run}_diff --export=ALL,REGION=cwb,CONFIG_NAME=exp_static2019_1_diff_gen_${run} \
  jobs/alex_eu_generate.slurm
# N8 (8 GPU):
sbatch --gres=gpu:a100:8 -J gen_${run}_diff \
  --export=ALL,REGION=cwb,CONFIG_NAME=exp_static2019_1_diff_gen_${run},NUM_GPUS=8 \
  jobs/alex_eu_generate.slurm
```
