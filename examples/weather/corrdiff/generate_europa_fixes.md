# Fixes to get `generate.py` running on Alex (Europa)

Notes from getting the first end-to-end Europa generation run (`diff_eu_v1` on the
2021 test split) to launch successfully on Alex with the zarr-v3 container.
Three files changed: `generate.py`, `jobs/alex_eu_generate.slurm`,
`conf/config_generate_europa-alex.yaml`.

## Symptoms (failure progression)

The same `sbatch jobs/alex_eu_generate.slurm` failed four times in a row, each
time uncovering a new defect once the previous one was fixed:

| Job ID    | Lifetime | First failing line in `generate.py` | Root cause                            |
| --------- | -------- | ----------------------------------- | ------------------------------------- |
| 3671755   | 1m 38s   | line 47 (`def main(cfg: DictConfig)`) | `DictConfig` / `OmegaConf` not imported |
| 3671892   | 49s      | line 53 (`DistributedManager.initialize()`) | distributed env vars unset            |
| 3677056   | 41s      | line 67 (`torch.as_tensor(seeds)…`)         | `torch` not imported                  |
| (would-be next) | —  | lines 87 / 107 / 135 / 302          | 4 more missing imports                |

After the third failure I stopped resubmitting blindly and ran a single AST
scan (below) that surfaced every remaining missing import at once. Job 3680880
then ran cleanly.

## Change 1 — `generate.py`: restore lost imports

The import block had been mangled in earlier hand edits. Each unimported name
was a latent `NameError` waiting for its code path to execute. Restored:

```python
import torch
from torch.distributed import gather
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
…
from helpers.train_helpers import set_patch_shape
from datasets.dataset import register_dataset
```

Also removed a stray duplicate `import netCDF4 as nc`.

Mapping (why each was needed):

| Symbol             | First use            | Source                          |
| ------------------ | -------------------- | ------------------------------- |
| `DictConfig`       | type annotation `main(cfg: DictConfig)` | `omegaconf`           |
| `OmegaConf`        | `.select`, `.to_container` on the cfg   | `omegaconf`           |
| `torch`            | 24 call sites throughout                | stdlib (3rd-party)    |
| `gather`           | bare `gather(image_out, …)` in the multi-rank path | `torch.distributed` |
| `to_absolute_path` | resolving checkpoint paths              | `hydra.utils`         |
| `set_patch_shape`  | patching setup branch                   | `helpers.train_helpers` |
| `register_dataset` | dataset registration before `get_dataset_and_sampler` | `datasets.dataset` |

### How to catch this class of bug fast (AST scan)

`import generate` only proves *module-level* names resolve — it does **not**
detect a missing import that is only referenced inside a function body (the
`torch` case). pyflakes / ruff / flake8 are not installed in the apptainer
image, so use this self-contained AST check:

```bash
apptainer exec ~/apptainer/corrdiff_zarr3.sif python3 - <<'PY'
import ast, builtins
src = open('generate.py').read()
tree = ast.parse(src)
bound = set(dir(builtins)) | {'__name__','__file__','__doc__','__package__'}
for n in ast.walk(tree):
    if isinstance(n, ast.Import):
        for a in n.names: bound.add((a.asname or a.name).split('.')[0])
    elif isinstance(n, ast.ImportFrom):
        for a in n.names: bound.add(a.asname or a.name)
    elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        bound.add(n.name)
    elif isinstance(n, ast.arg):
        bound.add(n.arg)
    elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
        bound.add(n.id)
    elif isinstance(n, (ast.Global, ast.Nonlocal)):
        bound |= set(n.names)
    elif isinstance(n, ast.ExceptHandler) and n.name:
        bound.add(n.name)
undef = {}
for n in ast.walk(tree):
    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in bound:
        undef.setdefault(n.id, n.lineno)
for k,v in sorted(undef.items(), key=lambda x:x[1]):
    print(f"line {v}: {k}")
PY
```

This over-approximates "defined" (it doesn't model scopes or use-before-assign)
but is exactly right for "name never bound anywhere in the file" — i.e. missing
imports. Run it after any non-trivial edit to the import block.

## Change 2 — `jobs/alex_eu_generate.slurm`: launch under `torchrun`, not bare `python3`

`DistributedManager.initialize()` in physicsnemo tries the torch-env path first
(needs `RANK` / `WORLD_SIZE` / `MASTER_ADDR`); on failure it falls through to
its SLURM path, which reads `SLURM_LAUNCH_NODE_IPADDR`. With bare
`apptainer run … python3 generate.py`, **neither** is set —

- torch env vars only exist when launched by `torchrun` (or `torch.distributed.run`),
- `SLURM_LAUNCH_NODE_IPADDR` is only populated inside an `srun` step, not
  inside a bare `apptainer run` invoked from the batch script.

The fix matches the working training jobs: wrap `generate.py` in `torchrun`
with a single process.

```bash
MASTER_PORT=$((20000 + SLURM_JOB_ID % 30000))

apptainer run --nv \
    --bind ${EUROPA_WS}:${EUROPA_WS} \
    --bind $WORK/checkpoints:/checkpoints/ \
    --bind $HPCVAULT/generated:$HPCVAULT/generated \
    $CONTAINER \
    torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
        $HOME/corrdiff/generate.py \
        --config-name config_generate_europa-alex
```

`--nproc_per_node=1` because the job is single-GPU by design (avoids
multi-rank NetCDF-writer coordination — see slurm header comment).

## Change 3 — `conf/config_generate_europa-alex.yaml`: `seed_batch_size: 8 → 16`

With `num_ensembles=16`, raising `seed_batch_size` from 8 to 16 collapses the
diffusion from 2 passes per timestamp to **1 pass per timestamp**. Confirmed at
runtime by the tqdm bar reading `1/1 [00:56<…]` instead of `2/2`.

`seed_batch_size` is bounded above by `num_ensembles`: setting it to 32 with
only 16 seeds crashes inside
`physicsnemo.utils.corrdiff.utils.StackedRandomGenerator.randn`, which
raises:

```
ValueError: Expected first dimension of size 16, got 32
```

(because `generate.py:272` expands `img_lr` to `seed_batch_size` rows but
`rank_batches` still holds only `num_ensembles` seeds.) To generate more
members, raise `num_ensembles` and leave `seed_batch_size ≤ num_ensembles`.

## Measured behavior on the 80 GB A100

From job 3680880, queried live via
`srun --overlap --jobid <ID> nvidia-smi --query-gpu=memory.used,…`:

| Metric             | Value             |
| ------------------ | ----------------- |
| GPU memory used    | **64 313 / 81 920 MiB** (~63 GB, ~79 %) |
| GPU util           | **100 %**         |
| Mem-bandwidth util | 60 %              |
| Per-timestamp wall | ~56 s (1 diffusion pass of 16 members) |
| Full run, 266 ts   | ~4.1 h (within the 6 h wall) |

Implications:

- **Batch 16 fits but leaves only ~17 GB headroom.** My earlier
  back-of-envelope estimate (~10–20 GB) was an undercount — PyTorch's caching
  allocator holds the diffusion sampler's peak reservation across the ~18 EDM
  steps, and `channels_last` adds layout overhead.
- **Batch 32 on this card would very likely OOM.** If you want 32 ensemble
  members, set `num_ensembles=32` and keep `seed_batch_size=16` (2 passes,
  ~2× wall time, same final ensemble).
- GPU is **compute-saturated at batch 16**, so increasing batch further would
  give little speedup even if memory allowed it.

## Verification checklist for future generate-config changes

1. Run the AST scan above on `generate.py` after any import edit.
2. Sanity-check the job script still launches `generate.py` under `torchrun`
   (not bare `python3`) — the distributed-init failure is silent until runtime.
3. Confirm `seed_batch_size ≤ num_ensembles`.
4. After submitting, peek at live GPU usage with
   `srun --overlap --jobid <ID> nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv`
   — `sstat` does not report GPU memory.
