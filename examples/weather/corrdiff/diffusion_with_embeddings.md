# Training the diffusion model with earth embeddings

How to train the CorrDiff **diffusion** stage with the satellite earth-embedding
inputs (Alpha/OLMO), given that the embedding integration was originally added and
tested on the **regression** stage. Companion to
[alpha-integrate.md](alpha-integrate.md) (which covers the dataset-side design).

Verified against the code in `train.py` + the physicsnemo classes inside
`corrdiff_zarr3.sif` on 2026-06-10.

---

## TL;DR

* **Concat is config-only.** The branch variant needed a **one-line registration fix**
  (see below) — discovered 2026-06-12 when all 4 branch diffusion runs failed at
  startup with `ValueError: Model type 'SongUNetEmbBranch' is not supported`.
* The embedding integration lives entirely in the **dataset**
  (`datasets/cwb.py`), which appends the embedding channels to the input. The model
  is agnostic, and `train.py` builds the diffusion model
  (`EDMPrecondSuperResolution`) with the **same `model_args` plumbing** as
  regression — including the `model_type: SongUNetEmbBranch` override
  ([train.py:341](train.py#L341), [train.py:390](train.py#L390)).
* **The one thing that must match:** `training.io.regression_checkpoint_path` must
  point to a regression checkpoint that was **trained with embeddings** (so its
  input-channel count equals the embedding-augmented conditioning). See
  ["Does the regression have to match?"](#does-the-regression-have-to-match) below
  — the answer is more permissive than "same integration".

---

## What to put in the diffusion config

Start from `conf/config_training_taiwan_diffusion-alex.yaml` and mirror the
`exp_static2019_1_reg_*` convention.

### Concat variant

Just enable embeddings in the `dataset:` block. `img_in_channels` grows
automatically from `dataset.input_channels()`; the default diffusion UNet
(`SongUNetPosEmbd`) ingests the extra channels with no model changes.

```yaml
dataset:
  data_path: /data/cwa_dataset.zarr
  embedding_path: /home/vault/b214cb/b214cb18/regrid2/alpha/zarr/gcs_2019_N1.zarr
  embedding_n: 1
```

### Branch variant

**Code fix (required, one line each in `train.py` and `generate.py`).** `train.py`
registered `SongUNetEmbBranch` only into `UNet._wrapped_classes` (the *regression*
precond wrapper), so regression-branch worked but diffusion-branch raised
`ValueError: Model type 'SongUNetEmbBranch' is not supported`. The diffusion wrapper
`EDMPrecondSuperResolution` has its **own** `_wrapped_classes` whitelist that also
needs the class:

```python
EDMPrecondSuperResolution._wrapped_classes = (
    EDMPrecondSuperResolution._wrapped_classes | {"SongUNetEmbBranch"}
)
```

Added next to the existing `UNet._wrapped_classes` line in **both** `train.py`
(for training) and `generate.py` (so emb-branch *diffusion* checkpoints load at
inference). Verified by instantiating `EDMPrecondSuperResolution(model_type=
"SongUNetEmbBranch", …)` and running a forward pass → `(B,4,448,448)`.

Config side — same `dataset:` block **plus** the same `model.model_args` used for the
regression branch run:

```yaml
model:
  hr_mean_conditioning: true        # already set in the diffusion base config
  model_args:
    N_grid_channels: 4
    embedding_type: zero
    model_type: SongUNetEmbBranch
    alpha_earth_channels: 64        # 64 alpha / 128 olmo
    emb_branch_channels: 64
```

And, as for the branch regression, **turn torch_compile off** (the branch model
re-implements `forward` and graph-breaks):

```yaml
training:
  perf:
    torch_compile: false
```

---

## Why the channel ordering works for branch + diffusion (verified)

The branch model splits the embedding off the **last** `alpha_earth_channels` of
its input ([models/song_unet_emb_branch.py](models/song_unet_emb_branch.py),
legacy-concat path `x[:, :-alpha_earth_channels]`). For diffusion the model input
is assembled by two concatenations, and both keep the embeddings trailing:

1. `EDMPrecondSuperResolution.forward` → `arg = cat([c_in * x, img_lr])`
   (noised target **first**, conditioning **last**).
2. Inside `ResidualLoss` with `hr_mean_conditioning=True` →
   `y_lr = cat((y_mean, y_lr))` — the regression mean is **prepended**, so the
   embedding channels stay last within the conditioning.

Net input to `SongUNetEmbBranch`:
`[ noised_target(4), y_mean(4), ERA5+grid…, embeddings(last 64) ]`
→ the trailing-channel split grabs exactly the embeddings. ✓

So concat-vs-branch styles can even be **mixed** between the regression and
diffusion stages: every regression net (concat or branch) consumes the identical
`[ERA5…, embeddings(last)]` conditioning tensor and only differs internally.

---

## Does the regression have to match?

The diffusion training pipeline feeds the **same conditioning tensor** (ERA5 **+
embeddings**) into the regression net to compute the hr_mean: inside `ResidualLoss`
it calls `regression_net(y_lr_res)` where `y_lr_res` is the full dataset
conditioning. So the real constraint is the regression net's **expected input
channel count**, not its architecture.

| Regression checkpoint | Works for embedding diffusion? |
|---|---|
| Trained **with** embeddings (concat) | ✅ directly — no edits |
| Trained **with** embeddings (branch) | ✅ directly — no edits (can even pair with a concat diffusion, and vice-versa) |
| Trained **without** embeddings | ❌ as-is — channel-count mismatch at `regression_net(...)` |

### So is it *impossible* without a matching regression? No.

A weather-only (no-embedding) regression + embedding-aware diffusion is possible
with a **small code edit**: slice the embedding channels off the conditioning
*before the regression call only*, so the regression net sees weather-only while
the diffusion model still sees weather + embeddings. This is arguably a cleaner
design — deterministic mean from weather, embedding signal carried only in the
stochastic residual.

Edit point: the regression-net call lives in physicsnemo's `ResidualLoss`
(in the container), so don't patch it in place. Instead subclass it (or extend the
existing `losses/emb_branch_losses.py:EmbResidualLoss` wrapper, which `train.py`
already selects when embeddings are separate) and override so that the tensor
passed to `self.regression_net(...)` is sliced to the weather channels, while the
tensor used as the diffusion conditioning keeps all channels. ~A few lines.

---

## Runtime requirements (same as the regression embedding runs)

* `--bind /home/vault:/home/vault` into apptainer (`$HPCVAULT` is **not**
  auto-mounted on Alex) so `embedding_path` resolves.
* Use the zarr-v3 container `corrdiff_zarr3.sif` (embedding stores are zarr v3).
* Do **not** pass `--cleanenv` / `PYTHONNOUSERSITE` (keeps zarr 3.x active).
* DDP `find_unused_parameters` is auto-set `False` for `cfg.model.name == diffusion`
  ([train.py:430](train.py#L430)), same as plain regression — fine for the branch
  model (all params are used). See `ddp_find_unused_params` memory if a future
  variant changes this.
