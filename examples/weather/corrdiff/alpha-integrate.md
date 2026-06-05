# Earth-embedding integration in CorrDiff (Alpha Earth / OLMO)

How regridded satellite embeddings (Alpha Earth, OLMO) are fed to the CorrDiff
models, and the reasoning behind the design. Written for future-us: it should be
possible to add OLMO, switch region/resolution, or change the integration style
from this document alone.

Status: **v2 (year-matched) implemented and smoke-tested 2026-05-31** against the
real Taiwan store + real Taiwan alpha embeddings. Supersedes v1 (single static
2024 store) — see [§ History](#history).

> **Current direction (2026-06): single-year via `embedding_path` is the primary
> interface.** Give one zarr file in `dataset.embedding_path` and it is appended to
> every sample (static, all years); the channel count is read from the file, and
> both the new `embedding` key and the legacy `alpha_earth` key are accepted.
> `embedding_path` takes precedence over the registry below and revives the old
> pre-registry configs. The per-year **registry / `v2_year`** machinery
> (`embedding_source` / `embedding_version` / `embedding_n` / `embedding_region` /
> `embedding_root`) is kept as a **dormant multi-year option** — to use it, leave
> `embedding_path: null` and set `embedding_source` + `embedding_version: v2_year`.
> The dataset **always logs an `EMBEDDINGS: ON/OFF` banner** at construction, and a
> configured-but-missing file is a hard error — embeddings are never silently
> skipped. The five `v2_year` run configs now live under `conf/multi_year/`
> (invoke as `--config-name multi_year/<name>`).

---

## TL;DR

* Per-year regridded embeddings are **appended as extra input channels**, matched
  to each sample's **own calendar year** (a 2020 weather sample sees the 2020
  embedding). Year-matching lives entirely in the dataset; the model is agnostic.
* Two integration styles share that one dataset path:
  * **`alpha-v2_year_concat`** — plain concat into the regression UNet input.
  * **`alpha-v2_year_branch`** — the trailing embedding channels are split off and
    processed by a parallel CNN (`SongUNetEmbBranch`) before fusion.
* Driven by four config knobs (`embedding_source`, `embedding_version`,
  `embedding_n`, `embedding_region`); channel count and filename are resolved by a
  registry in [`datasets/cwb.py`](datasets/cwb.py). No filename patterns in config.
* **No file restructuring** — files stay at `$HPCVAULT/regrid2/<source>/zarr/`.

---

## The regridded embedding files

Location: `$HPCVAULT/regrid2/` = `/home/vault/b214cb/b214cb18/regrid2/`.
One **static annual** store per (source, region, year, N), already on the exact
CWA/Europa target grid. Both `.nc` (netCDF4) and `.zarr` (zarr **v3**) versions
exist with identical content; the loader reads the `.zarr`.

| source | prefix | channels | taiwan file | europa file | years present |
|--------|--------|----------|-------------|-------------|---------------|
| alpha  | `gcs`  | 64       | `alpha/zarr/gcs_{year}_N{n}.zarr`  | `gcs_eu_{year}_N{n}.zarr`  | 2018-2021, 2024 (taiwan); **2018, 2021 only** (europa) |
| olmo   | `olmo` | 128      | `olmo/zarr/olmo_{year}_N{n}.zarr` | `olmo_eu_{year}_N{n}.zarr` | 2018-2021 |

* Array key inside every store: **`embedding`**, shape `(C, 450, 450)` float32,
  with global attrs `source`, `year`, `N`.
* **N** = embedding pixels per weather pixel. **N1** = same 450×450 grid as the
  weather store (what we use). N8 = 3600×3600 (8× finer) — needs a downsampling /
  encoder path and is **out of scope** (real model-dim change).
* ~70 % of each field is NaN (ocean, for Taiwan) → filled to **0** on load.
* Embeddings are **annual and static**: every hourly sample in a year gets the
  same field. Year-to-year change is modest (land-cover/use), so year-matching is
  primarily a *correctness* fix over v1's fixed-2024 store, not a guaranteed large
  metric gain — but it is cheap and removes the train/embedding year mismatch.

---

## Code: registry + loader ([datasets/cwb.py](datasets/cwb.py))

Module-level registry (top of file) maps source → (prefix, channels) and region →
filename token, and builds the store path:

```python
_EMB_SOURCES = {"alpha": ("gcs", 64), "olmo": ("olmo", 128)}
_EMB_REGION_TOKEN = {"taiwan": "", "europa": "eu"}
def _emb_store_path(root, source, region, year, n):  # -> .../alpha/zarr/gcs_2020_N1.zarr
def _load_embedding_store(path, img_y, img_x):        # open, NaN->0, crop to 448, -> tensor
```

`ZarrDataset.__init__` params (defaults): `embedding_source="none"`,
`embedding_version="v2_year"`, `embedding_n=1`, `embedding_region="taiwan"`,
`embedding_root=None` (must be set per run config; the loader raises if embeddings
are enabled and it is `None`), `embedding_static_year=2024`.

How the **year for each sample** is determined and matched (the crux):

1. `self._dataset` is the already-split dataset (`FilterTime(is_not_2021)` for
   train, `FilterTime(is_2021)` for val), so its index order *is* the sample order.
2. In `__init__` (`v2_year`): `self._sample_years = [t.year for t in self._dataset.time()]`
   — index-aligned years — then load `self._year_emb[year]` for each unique year
   present (only the split's years; train loads `{2018,2019,2020}`, val `{2021}`).
   Missing year file → `FileNotFoundError`.
3. In `__getitem__(idx)`: append `self._year_emb[self._sample_years[idx]]` to the
   input. Because `FilterTime.time()` and `FilterTime.__getitem__` walk the same
   `_indices`, the year at position `idx` always belongs to the data at `idx`.

`v1_static` instead loads one `embedding_static_year` store into `self._static_emb`
and appends it to every sample (reproduces old v1). Unknown `embedding_version`
raises (so a future bump is explicit). `input_channels()` appends
`self.embedding_channels` entries named `{source}_emb_{i}`, so the model's
`img_in_channels` grows automatically via `train.py`. The deprecated
`embedding_path` kwarg is accepted-but-ignored (with a warning) so stale configs
don't crash.

---

## The two integration styles (model side)

The dataset appends the same year-matched channels in both cases; only the model
differs. **Both** styles get year-matching for free.

### `alpha-v2_year_concat` — plain concat (regression UNet)
The 64 alpha channels are stacked next to the 12 ERA5 channels; the UNet's input
stem (now 76→128) processes them together from step 0. Simplest; the standard
CorrDiff way of adding static fields (orography, land-sea mask). Best **baseline**
for the question "do these embeddings carry signal at all?".

### `alpha-v2_year_branch` — parallel CNN ([models/song_unet_emb_branch.py](models/song_unet_emb_branch.py))
`SongUNetEmbBranch` (named after Yang Song's NCSN++/DDPM++ diffusion U-Net, the
EDM "Song U-Net") does **not** feed the embeddings into the main UNet input. It
processes them in a separate branch and joins them deeper in.

#### In simple words — what connects to what, and in what shape

Think of each "channel" as a **448×448 sheet** of numbers; "connecting" two stacks
means **putting their sheets on top of each other** (concatenate along channels).

* **Plain concat (v1.1):** stack the *raw* sheets at the very input —
  `(12 ERA5) + (64 embedding) = (76, 448, 448)` → into the UNet stem. One step,
  at the start.
* **emb_branch:** keep them apart, process each, join *deeper*:

```
INPUT (80, 448, 448) = 4 placeholder + 12 ERA5 + 64 embedding
        │
        ├── SPLIT ────────────────┬───────────────────────────────┐
        │                         │                               │
   x_main (16,448,448)       x_emb (64,448,448)                   │
   placeholder + weather      the embeddings                      │
        │                         │                               │
   MAIN UNET                  EMBEDDING BRANCH (mini-CNN)          │
   + grid → (20,448,448)      Conv3x3 → GroupNorm → SiLU          │
   stem conv → (128,448,448)  Conv3x3 → GroupNorm → SiLU          │
   level-0 blocks (128,448,448)   → emb_feat (64,448,448)         │
        │                         │                               │
        └──────────┬──────────────┘                               │
                   │  CONNECT = stack the sheets                   │
        concat → (192,448,448)   [128 main + 64 embedding]        │
        1x1 conv squeeze → (128,448,448)  (per-pixel blend)       │
                   │                                               │
              REST OF U-NET: downsample 448→224→… → out (4,448,448)
```

The join happens **once, at full 448×448 resolution, right before the first
downsample** ([:179-181](models/song_unet_emb_branch.py#L179-L181)). The branch is
2× (`Conv 3×3 → GroupNorm(32) → SiLU`, [:84-91](models/song_unet_emb_branch.py#L84-L91)),
keeping 64 channels at 448×448 (a 5×5 receptive field — local features only). The
fusion is `concat(128, 64)=192` → **1×1 conv** back to 128
([:96-104](models/song_unet_emb_branch.py#L96-L104)) — a per-pixel learned blend,
no spatial mixing.

#### Near-zero init — honest caveat
The fusion 1×1 conv is initialized near zero (`init_weight=1e-5`). The docstring
says this keeps training "close to baseline at init", but the fusion **replaces**
`x` (`x = fusion_conv(cat([x, emb_feat]))`, not `x = x + …`), so at init it
near-zeros the feature map going into the *downsampling* path — only the level-0
**skip connection** (captured before the join) preserves the main signal. So the
deep encoder/decoder starts ~silent and "warms up" in the first steps; it is not
literally the baseline. A **residual** fusion (`x = x + fusion_conv(...)`) would
match the stated intent better — open change if the branch trains slow to start.

**Footgun:** `model_args.alpha_earth_channels` must equal the source channel count
(64 alpha, **128 olmo**); manual knob, misbehaves silently if wrong. This class also
re-implements `SongUNet.forward`'s encoder/decoder loop, so it can drift from the
parent as the base model evolves (and torch.compile tends to graph-break on it).

Opinion: year-matching matters more than concat-vs-branch (same information either
way; architecture only changes how easily it's learned). Run both, compare against
the no-embedding checkpoint on the same val set. Start with concat for the clean
signal test; reach for the branch to squeeze more.

---

## Config knobs

Defaults live in the dataset **base** config ([conf/base/dataset/cwb.yaml](conf/base/dataset/cwb.yaml),
europa.yaml); a run config overrides the per-experiment ones:

```yaml
embedding_source: none        # none | alpha | olmo   (none = off)
embedding_version: v2_year    # v1_static (single fixed year) | v2_year (year-matched)
embedding_n: 1                # 1 | 8   (start with 1)
embedding_region: taiwan      # taiwan | europa   (europa.yaml overrides to europa)
embedding_root: null          # MUST be set in each run config that enables embeddings
embedding_static_year: 2024   # only used when embedding_version: v1_static
```

* **`embedding_root` is NOT hardcoded in base/code.** Base sets it to `null` and the
  Python default is `None`; the loader **raises** if `embedding_source != none` and
  `embedding_root is None`. Each run config that enables embeddings sets it explicitly
  in its `dataset:` block, e.g. `embedding_root: /home/vault/b214cb/b214cb18/regrid2`.
  This keeps the cluster-specific absolute path out of the shared base config and
  visible per run. (On another cluster, point it at that machine's regrid root, or
  use `${oc.env:HPCVAULT}/regrid2`.)
* **`embedding_region` is explicit, not inferred.** Both regions share `type: cwb`
  → `cwb.get_zarr_dataset`, distinguished only by `normalization`/`data_path`.
  Inferring region from `normalization` was rejected as too implicit; the explicit
  flag (defaulted per dataset base config) is mildly redundant but clearer.
* **`embedding_version` is a bumpable implementation version**, the hook for future
  integration changes — add `v3_…` and a new branch in `__init__`.

### Files added/changed

| File | Purpose |
|---|---|
| [datasets/cwb.py](datasets/cwb.py) | registry, `_emb_store_path`, `_load_embedding_store`, `ZarrDataset` init/getitem/input_channels, `get_zarr_dataset` |
| [conf/base/dataset/cwb.yaml](conf/base/dataset/cwb.yaml) | new `embedding_*` keys (Taiwan defaults) |
| [conf/base/dataset/europa.yaml](conf/base/dataset/europa.yaml) | same keys, `embedding_region: europa`, off by default |
| [conf/multi_year/config_training_taiwan_regression-alpha-v1.1.yaml](conf/multi_year/config_training_taiwan_regression-alpha-v1.1.yaml) | concat training, ckpt `/checkpoints/taiwan/alpha-v2_year_concat` |
| [conf/multi_year/config_training_taiwan_regression-emb_branch-year.yaml](conf/multi_year/config_training_taiwan_regression-emb_branch-year.yaml) | branch training, ckpt `/checkpoints/taiwan/alpha-v2_year_branch` |
| [conf/multi_year/config_generate_taiwan-alpha-v1.1.yaml](conf/multi_year/config_generate_taiwan-alpha-v1.1.yaml) | concat generation (test-2021 split → vault) |
| [conf/multi_year/config_generate_taiwan-emb_branch-year.yaml](conf/multi_year/config_generate_taiwan-emb_branch-year.yaml) | branch generation |
| [jobs/multi_year/alex_taiwan_alpha_v1.1.slurm](jobs/multi_year/alex_taiwan_alpha_v1.1.slurm) | train concat (4×a100, torch_compile on) |
| [jobs/multi_year/alex_taiwan_emb_branch_year.slurm](jobs/multi_year/alex_taiwan_emb_branch_year.slurm) | train branch (compile off — custom forward) |
| [jobs/multi_year/gen_alex_taiwan_alpha_v1.1.slurm](jobs/multi_year/gen_alex_taiwan_alpha_v1.1.slurm) | generate concat (1×a100, torchrun) |
| [jobs/multi_year/gen_alex_taiwan_emb_branch_year.slurm](jobs/multi_year/gen_alex_taiwan_emb_branch_year.slurm) | generate branch |

The SLURM scripts follow the modern Europa convention (zarr-v3 container,
`corrdiff_zarr3.sif`; unique `MASTER_PORT`; `el9` constraint) and add the two
Taiwan-specific binds: the relocated CWA store → `/data`, and `/home/vault` for
the embeddings. Generation needs **no code change**: the architecture is restored
from the checkpoint (`Module.from_checkpoint`), and `get_zarr_dataset` computes
each requested timestamp's year and loads the matching store automatically.

---

## Runtime requirements (important)

* **Bind the embeddings into the container.** `$HPCVAULT` (`/home/vault`) is **not**
  auto-mounted in apptainer on Alex. Add `--bind /home/vault:/home/vault` to the
  training/generation `apptainer` commands, or `embedding_root` won't resolve.
* **zarr 3.x must be active.** The embedding stores are zarr v3. `~/.local` has
  zarr 3.1.5 which shadows the container's 2.18 and also reads the v2 CWA store.
  Do **not** pass `--cleanenv` / `PYTHONNOUSERSITE`, or the v3 stores become
  unreadable. (The europa container `corrdiff_zarr3.sif` also works.)
* **Dataset paths on Alex** (workspaces, not `/data`): the Taiwan store currently
  lives at
  `/anvme/workspace/b214cb18-ws-daniel3/b214cb18-ws-daniel-cwb-1779073861/cwa_dataset/cwa_dataset.zarr`
  (the old `ws-daniel-cwb` workspace expired; job scripts that hardcode it need
  updating). Configs use `/data/cwa_dataset.zarr` via a 1:1 bind.

---

## Verification

Smoke-tested 2026-05-31 in `corrdiff_10_02.sif` with `--bind /home/vault --bind /anvme`
and `PYTHONPATH=…/corrdiff` (zarr 3.1.5), against the real Taiwan store + alpha files:

* **Per-year load**: all four years' stores load to `(64, 448, 448)` and differ.
* **Train split**: `len(input_channels()) == 76` (12 ERA5 + 64), loads exactly
  `{2018,2019,2020}`, channel names `alpha_emb_*`, input shape `(76,448,448)`, and
  each sample's appended block **equals its own year's store** (checked a 2018 and
  a 2019 sample).
* **Val split**: loads only `{2021}` and matches the 2021 store.

To re-run, build the dataset directly:
```python
from datasets.cwb import get_zarr_dataset
ds = get_zarr_dataset(data_path=<taiwan.zarr>, normalization="v1", train=True,
                      out_channels=(0,1,2,3),  # this store has 4 cwb channels
                      embedding_source="alpha", embedding_version="v2_year")
assert len(ds.input_channels()) == 76
```
(Default `out_channels=(0,17,18,19)` is for the 20-channel CWA store; this store
has 4 cwb channels, so pass `(0,1,2,3)` as the configs do.)

Still to do by the user: short `test`-partition training run per config, and a
generation smoke run on a 2020 + 2021 timestamp.

---

## Extending later (enabled by the knobs)

* **OLMO**: `embedding_source: olmo` → 128 channels, `in_channels` becomes 12+128.
  For the branch model also set `model_args.alpha_earth_channels: 128`.
* **Europa**: `embedding_region: europa` (already defaulted in europa.yaml). Note
  only 2018 & 2021 alpha-eu files exist; `v2_year` on a 2018-2020 train split will
  raise until 2019/2020 are regridded.
* **N8**: not just a config flip — 3600×3600 needs a downsampling/encoder stage.
  **Done** — implemented as a separate-tensor + `pixel_unshuffle(8)` branch
  front-end; see [n8-integrate.md](n8-integrate.md).

---

## History

* **v1 (superseded)**: a single static `2024-conservative.zarr` (key `alpha_earth`)
  concatenated to *every* sample regardless of date — temporally mismatched, one
  year only. Config key was `dataset.embedding_path`. Now removed; the loader reads
  the per-year registry instead. `embedding_path` is accepted-but-ignored so old
  configs (`config_training_taiwan_regression-alex.yaml`, `-emb_branch`,
  `config_generate_taiwan.yaml`) don't crash, but they no longer load embeddings.
* **v2 (current)**: year-matched per-source/region/N registry; both concat and
  branch styles; OLMO/Europa-ready.
