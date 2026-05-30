# Training Optimizations — Europa CorrDiff Regression (Alex cluster)

Notes from the 2026-05-21/22 optimization pass. **These are local changes — not upstream PhysicsNeMo defaults.** Every code edit is marked with `# === custom edit ===` so they're greppable.

## Summary

| | Before | After |
|---|---|---|
| Step time | 22.8 sec/tick | **16.0 sec/tick** (−30%) |
| Peak GPU memory (steady) | 52 GB | **24 GB** (−55%) |
| GPU type | Required `--constraint=a100_80` | Works on **40 GB A100** too |
| GPU count | 2 | **4** (recommended for full 2M run) |
| Validation logging | aggregate only | aggregate + **per-channel** |
| Extra wandb keys | — | `perf/{samples_per_sec, peak_gpu_mem_gb, grad_norm}`, `val/loss_<channel>` |

For the resume run targeting 2M samples (from checkpoint 400k): **one ~15 h job** instead of **two ~14 h jobs**.

---

## Code changes (`train.py`)

All five edits marked with `# === custom edit ===` headers in the file.

### 1. Conditional `find_unused_parameters` ([train.py:380](../train.py#L380))
Was: hardcoded `True` (PhysicsNeMo's defensive default for all variants).
Now: `False` for `regression` and `diffusion` (fully static graphs), `True` for everything else.
- Saves ~1-3% per step.
- **Footgun**: switching to `lt_aware_ce_regression`, `patched_diffusion`, or `lt_aware_patched_diffusion` requires flipping back to `True` or DDP hangs at iter 0-1. (See `~/.claude/projects/-home-hpc-b214cb-b214cb18-corrdiff/memory/ddp_find_unused_params.md`.)

### 2. Validation uses uncompiled model ([train.py:766](../train.py#L766))
```python
eval_net = getattr(model, "_orig_mod", model)
loss_valid_kwargs = {"net": eval_net, ...}
```
- `torch.compile` recompiles whenever `grad_mode` toggles (e.g. entering `with torch.no_grad():`). Without this, every validation event triggers ~15 min of Inductor recompile and the GPU sits idle at ~100W.
- `_orig_mod` is the DDP-wrapped model without the compile layer. `getattr(..., model)` falls back gracefully when compile is disabled.

### 3. Per-channel validation loss ([train.py:790-820](../train.py#L790))
Captures the loss tensor (shape `(B, C, H, W)`) before reduction, computes `mean(dim=(0,2,3))` → per-channel per-pixel MSE, accumulates across `validation_steps`, all-reduces across ranks, logs as `val/loss_<channel>` to wandb.
- Shape guard (`if loss_valid.ndim == 4`) silently skips per-channel for loss tensors with unexpected shapes; aggregate `validation_loss` still works.
- Values in **z-scored space** (e.g. T2 loss = 0.1 ≈ RMSE 0.32 std ≈ ~3 °C for European temperature).

### 4. Gradient norm capture ([train.py:705](../train.py#L705))
```python
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")).item()
```
- One-line addition after `handle_and_clip_gradients`. `inf` threshold = compute norm without clipping.
- No helper-function changes needed.

### 5. Perf metrics in periodic wandb.log ([train.py:881-895](../train.py#L881))
Added `perf/samples_per_sec`, `perf/peak_gpu_mem_gb`, `perf/grad_norm` to the existing wandb.log dict in the print block.
- `peak_mem_gb` is captured **before** `reset_peak_memory_stats()` so the wandb value reflects the real peak, not zero.

---

## Config changes (`conf/config_training_europa_regression-alex.yaml`)

| Setting | Old | New |
|---|---|---|
| `training.hp.training_duration` | `1_000_000` | `2_000_000` |
| `training.io.save_n_recent_checkpoints` | `5` | `15` |
| `training.perf.torch_compile` | (not set, defaults False) | `true` |
| `wandb.resume_id` | `null` | `xzl9vnzo` (continues existing run) |

---

## SLURM changes (`jobs/alex_europa_regression.slurm`)

| Setting | Old | New |
|---|---|---|
| `--gres=gpu:a100:N` | `2` | `4` |
| `NUM_GPUS` | `2` | `4` |
| `--constraint` | `a100_80` | `el9` (any VRAM, newer OS) |
| `torchrun --master_port=...` | (default 29500) | `$((20000 + SLURM_JOB_ID % 30000))` |

**Why no `a100_80`**: peak GPU memory at bs=4 + compile is only 33 GB (compile-time) / 24 GB (steady), well under 40 GB.

**Why MASTER_PORT**: Alex shares physical compute nodes between jobs. When two multi-GPU jobs land on the same node, both default torchruns try to bind port 29500 → second fails with `EADDRINUSE`. Deriving the port from `SLURM_JOB_ID` guarantees uniqueness. (Julia2 doesn't have this issue because it gives whole-node allocations for GPU jobs.)

---

## Test findings

### torch.compile vs baseline (bs=4, 2 GPUs, 80 GB)

| samples | baseline loss | compiled loss | Δ | baseline sec/tick | compiled sec/tick | baseline peak_mem | compiled peak_mem |
|---|---|---|---|---|---|---|---|
| 4096 | 488,172 | 487,408 | −0.16% | 22.8 | 16.0 | 59.9 GB | 32.9 GB (compile spike) |
| 8192 | 154,676 | 156,343 | +1.08% | 22.8 | 16.0 | 52.6 GB | 23.9 GB |
| 12032 | 103,321 | 103,745 | +0.41% | 22.8 | 16.0 | 52.6 GB | 23.9 GB |

- Loss values within ~1% across all checkpoints — well below the run's own tick-to-tick noise (4-15%). torch.compile **does not change training math** meaningfully.
- Compile overhead: ~5 min (96 sec extra to reach first sample log). Break-even at ~7,680 samples.
- Memory dropped 55% — fused kernels eliminate intermediate tensor allocations.

### Batch size scaling (with compile, 2 GPUs)

| bs_per_gpu | accum rounds | train sec/tick | val sec/tick | peak_mem |
|---|---|---|---|---|
| 4 | 32 | 16.0 | 19.1 | 24 GB |
| 8 | 16 | 15.9 | **21.6** | 47 GB |

- **bs=8 gave essentially no training speedup** (0.6%, within noise) because torch.compile already absorbed per-round overhead.
- Validation became **13% slower** (eager-mode forward on 2× larger batches; `_orig_mod` workaround means val isn't compiled).
- Over a full 1M sample run, bs=8 is **~8 min slower** total. bs=4 wins.
- bs=12 estimated to fit (~70 GB) but worse for the same reason. bs=16 doesn't fit (~93 GB).

### 40 GB vs 80 GB A100

| GPU | Peak mem (first tick) | Peak mem (steady) | sec/tick |
|---|---|---|---|
| 80 GB | 32.88 GB | 23.94 GB | 16.0 |
| 40 GB | 32.88 GB | (expected ~24 GB) | 16.9 (first tick only) |

- 40 GB fits with **7 GB margin** during the compile-time peak, 16 GB margin steady.
- Same speed as 80 GB (no fundamental reason it should be different — same compute, same HBM bandwidth).
- About half of Alex's a100 pool is 40 GB → wider candidate pool → faster scheduling.

### Multi-GPU scaling (estimated; only 2 GPUs measured)

| GPUs | accum rounds | sec/tick | samples/sec | Time for 1.6M remaining |
|---|---|---|---|---|
| 2 | 32 | 16.0 (measured) | 16 | ~28 h (needs 2 jobs) |
| 4 | 16 | ~8.5 (est) | ~30 | **~15 h** (one job) |
| 8 | 8 | ~4.8 (est) | ~53 | ~9 h (one job, but queue wait is long) |

- 4 GPUs hits the sweet spot: ~2× faster, fits one 23 h job, doesn't monopolize a node.
- 8 GPUs needs a whole Alex node — queue wait can be hours to days.

---

## Known cosmetic warnings (safe to ignore)

- **`Grad strides do not match bucket view strides`** — DDP buckets expect contiguous layout, compile emits channels-last for conv weights. ~1-3% perf hit. Fixable with `model.to(memory_format=torch.channels_last)` but not worth the risk for the marginal gain.
- **`[__graph_breaks]` Dynamo logs** — info, not errors. Coming from `train.py:70` which enables `torch._logging.set_logs(graph_breaks=True)`. Silence by setting `False` if too noisy.
- **`Pandas requires bottleneck >=1.3.6`** — container has 1.3.5. Cosmetic.
- **`TensorFloat32 tensor cores ... not enabled`** — for fp32 matmuls. We use bf16 so this hint doesn't apply to us.

---

## Pitfalls to remember

1. **Model variant switch**: if you change `model.name` to a patched/CE variant, flip `find_unused_parameters` back to `True` at [train.py:380](../train.py#L380).
2. **Removing `torch.compile` while keeping `_orig_mod` validation code**: the `getattr(..., model)` fallback handles this gracefully — no edit needed.
3. **`MASTER_PORT` derivation**: keep it. Without it, you'll randomly hit port collisions when SLURM puts two multi-GPU jobs on one node.
4. **Effective batch size**: changing `batch_size_per_gpu` while keeping `total_batch_size` constant changes nothing about training math (same gradients via accumulation). Only changing `total_batch_size` affects optimizer behavior and would need an LR retune.
