# N8 high-resolution embeddings in the emb-branch model

Companion to [alpha-integrate.md](alpha-integrate.md). That doc covers the N1
embeddings (one embedding pixel per weather cell) and the parallel-CNN
`SongUNetEmbBranch`. This doc covers the **N8** extension: embeddings at **8× the
weather resolution** (each 450-grid cell split into an 8×8 block of sub-pixels).

## TL;DR

- N8 stores are `(C, 3600, 3600)` — 8·450 per axis — vs N1's `(C, 450, 450)`.
  OLMO = 128 channels, Alpha = 64. Example: `olmo_2019_N8_masked.zarr`.
- N8 **cannot** be concatenated onto the 448×448 weather input (resolution
  mismatch), so it is delivered to the model as a **separate tensor** and reduced
  inside the branch by `pixel_unshuffle(8)` — which folds each cell's 8×8
  sub-pixels into channels (C → C·64) at the 448 grid, so a conv can **learn** the
  intra-cell variation instead of averaging it away.
- **Single source of truth:** the dataset's `embedding_n` (1 or 8) drives
  everything. `n>1` auto-enables the separate-tensor delivery and sets the model's
  `pixel_unshuffle` factor. `n==1` keeps the unchanged N1 concat path.
- Verified end-to-end on real data (job 3701158): trains, loss decreases, saves a
  checkpoint. **~36 GB GPU at bs=1**, `dataloader_workers: 0` required.

## Why a separate tensor (not concat, not averaging)

| Option | Verdict |
|---|---|
| Average 8×8 → 450 in the loader, concat as N1 | Throws away exactly the sub-cell signal we want — no better than N1. ✗ |
| `pixel_unshuffle` 8× → `(8192, 450, 450)`, concat into `img_lr` | Lossless, but rides in the weather input (8192+12 channels) and the loss copies a 6.6 GB block. ✗ (rejected) |
| **Separate tensor + `pixel_unshuffle` inside the branch** | Lossless, weather pipeline stays clean, embedding stays architecturally separate. **✓ chosen** |

`pixel_unshuffle(8)` is a pure rearrange: the 64 sub-pixels of each cell become 64
channels at the same grid point. A conv over those channels sees every sub-pixel
and learns whether/how they vary. `factor=1` is the identity → the N1 branch is
unchanged.

## Code changes

All changes are backward-compatible: with `embedding_n: 1` (or no embeddings) every
path behaves exactly as before.

### `datasets/cwb.py`
- `_emb_store_path(..., masked=False)` — appends `_masked` for the masked store
  variant (e.g. `olmo_2019_N8_masked.zarr`).
- `_load_embedding_store(..., n=1)` — crops to `(C, n·448, n·448)` (= 3584 for N8)
  instead of forcing 448; **no averaging/downsampling**; NaN→0 as before.
- `ZarrDataset.__init__` — new args `embedding_masked` and `embedding_separate`.
  `embedding_separate=None` (default) **auto-derives** to `embedding_n > 1`. Stores
  `self.embedding_n` / `self.embedding_separate`.
- `__getitem__` — in separate mode returns a 3-tuple `(target, input, embedding)`
  with the embedding **not** concatenated onto `input`; otherwise unchanged
  (legacy concat).
- `input_channels()` — excludes the embedding channels from the count in separate
  mode (they are no longer part of `input`).

### `models/song_unet_emb_branch.py`
- New `__init__` args `emb_downscale_factor` (the `pixel_unshuffle` factor) and
  `embedding_separate`. The branch's first conv is sized
  `alpha_earth_channels · factor²` (8192 for OLMO N8).
- Guard: `emb_downscale_factor > 1` requires `embedding_separate=True`
  (a higher-res embedding cannot be concatenated).
- In `embedding_separate` mode, `__init__` does **not** subtract the embedding
  channels from the main `in_channels` (the embedding no longer rides in `x`).
- `forward(..., embedding=None)` — in separate mode takes `x` as the full main
  input and the embedding from the kwarg, applies `pixel_unshuffle(factor)`
  (no-op when `factor==1`), then runs the existing branch + fusion unchanged.

### `losses/emb_branch_losses.py` (new)
The physicsnemo `RegressionLoss` / `ResidualLoss` have fixed signatures and can't
forward a separate tensor. These subclasses inject the embedding into the
`net(...)` calls via a thin `_EmbInjector` wrapper (both preconditioners already
forward `**model_kwargs` to the model). No container rebuild needed. With
`embedding=None` they behave exactly like the base classes.
- `EmbRegressionLoss` — regression path (primary use).
- `EmbResidualLoss` — diffusion path; injects into both the diffusion `net` and the
  frozen `regression_net`. **Patching is not yet supported** with a separate
  high-res embedding (would need patch-aligned cropping) — raises a clear error.

### `train.py`
- Reads the dataset's `embedding_separate` once; unpacks the 3-tuple, moves the
  embedding to device, and passes `embedding=` into the loss kwargs (train + val).
- For `model_type == "SongUNetEmbBranch"`, **injects** `emb_downscale_factor =
  embedding_n` and `embedding_separate = (embedding_n > 1)` into `model_args`, so
  these are persisted in the checkpoint (needed at generation) and can never
  desync from the dataset.
- Swaps in `EmbRegressionLoss` / `EmbResidualLoss` when separate mode is active.

### `generate.py`
- Unpacks the embedding from the loader and wraps `net_reg` / `net_res` with
  `_EmbInjector` (the diffusion net gets the embedding expanded to the seed batch)
  so `regression_step` / `diffusion_step` reach the model with `embedding=`.

### Config
- `conf/base/dataset/cwb.yaml` — added `embedding_masked` (default false) and
  `embedding_separate` (default **null = auto from `embedding_n`**).
- `conf/config_training_taiwan_regression-emb_branch-year-N8.yaml` — production
  OLMO N8 run config.
- `conf/config_training_taiwan_regression-emb_branch-year-N8-TEST.yaml` — smoke
  test (single static 2019 field, a few steps).
- `jobs/multi_year/alex_taiwan_emb_branch_N8_TEST.slurm` — test job.

> Note: these configs live at `conf/` **root**, not `conf/multi_year/`. The bare
> `- val_times_2021@_val_times` default only resolves from the config root; the
> `multi_year/` siblings (incl. the existing alpha emb_branch config) silently fail
> to compose for this reason.

## How to use it — one knob

In the dataset config set only:
```yaml
dataset:
    embedding_source: olmo      # olmo (128ch) | alpha (64ch)
    embedding_n: 8              # 1 = N1 (concat) | 8 = N8 (separate + pixel_unshuffle)
    embedding_masked: true      # *_masked store variant
    embedding_region: taiwan    # taiwan | europa
    embedding_root: /home/vault/b214cb/b214cb18/regrid2
```
and in the model set the channel count to match the source:
```yaml
model:
    model_args:
        model_type: "SongUNetEmbBranch"
        alpha_earth_channels: 128   # OLMO = 128, Alpha = 64
        emb_branch_channels: 64
        # emb_downscale_factor + embedding_separate are auto-injected from embedding_n
```
`embedding_version` selects `v1_static` (one fixed year) or `v2_year` (year-matched,
multiple resident years).

## Runtime requirements (measured, job 3701158)

| Resource | Value |
|---|---|
| GPU (bs=1) | **peak ~36 GB torch / ~37 GB nvidia-smi**. 80 GB A100 comfortable; 40 GB fits but only ~3 GB headroom (risky for long runs / DDP / cuDNN algo variation). |
| Batch size | `batch_size_per_gpu: 1`. bs=2 only on 80 GB. Each N8 sample is ~6.6 GB at the branch input. |
| CPU RAM | ~40 GB with one static year; budget ~60 GB for `v2_year` (3 resident years). |
| `dataloader_workers` | **0 (critical)** — workers fork-copy the resident ~6.6 GB/year fields. |
| `torch_compile` | **off** — `SongUNetEmbBranch` rewrites `forward()` (Python loop + checkpointing), which graph-breaks; same as the N1 branch config. (The plain/concat models keep compile on.) |
| Speed | ~1.7 s/sample at bs=1 steady state. |

## Verification

Smoke test (in container, `--bind /home/vault:/home/vault`, `corrdiff_zarr3.sif`):
- `_load_embedding_store(olmo_2019_N8_masked, n=8)` → `(128, 3584, 3584)`;
  `pixel_unshuffle(8)` → `(8192, 448, 448)`.
- `SongUNetEmbBranch` forward, all paths: separate factor=8, separate factor=1,
  legacy concat N1, missing-embedding guard, real 128→8192 width.
- Guard: `emb_downscale_factor>1` with `embedding_separate=False` raises.
- Full training job (1× A100-80, bs=1, `v1_static` 2019): **Training Completed**,
  loss varies, checkpoint saved, ~36 GB GPU.

## Not yet done / caveats

- **Diffusion + patching** with a separate N8 embedding is unsupported (the
  embedding would need patch-aligned cropping at 8× resolution) — `EmbResidualLoss`
  raises if `patching` is set with a separate embedding.
- The 40 GB A100 fit is tight; prefer 80 GB for production.
- Production config currently uses `v2_year`; switch to `v1_static` +
  `embedding_static_year` if a single static year is intended.
