"""
Generate the 24 exp_static2019_1_diff Hydra configs (12 train + 12 generate)
from the exp_static2019_1_reg run matrix.

Each diffusion train config mirrors its regression sibling but swaps
model=diffusion, adds hr_mean_conditioning + the matching embedding-trained
regression_checkpoint_path. Each generate config runs the full reg+diffusion
pipeline (16 ensembles, held-out test split, save_input: false).

Idempotent: rewrites the config files in conf/. Run from the repo root.
"""
from pathlib import Path

CONF = Path(__file__).resolve().parents[2] / "conf"

# Per-run matrix. embedding_path is hardcoded (olmo uses *_masked, europa uses
# the eu token, N8 uses the N8 store) — no clean formula, so it is explicit.
RUNS = {
    "taiwan_alpha_concat_N1": dict(region="cwb", embed="alpha", arch="concat", n=1,
        emb="/home/vault/b214cb/b214cb18/regrid2/alpha/zarr/gcs_2019_N1.zarr", ch=64),
    "taiwan_alpha_branch_N1": dict(region="cwb", embed="alpha", arch="branch", n=1,
        emb="/home/vault/b214cb/b214cb18/regrid2/alpha/zarr/gcs_2019_N1.zarr", ch=64),
    "taiwan_olmo_concat_N1": dict(region="cwb", embed="olmo", arch="concat", n=1,
        emb="/home/vault/b214cb/b214cb18/regrid2/olmo/zarr/olmo_2019_N1_masked.zarr", ch=128),
    "taiwan_olmo_branch_N1": dict(region="cwb", embed="olmo", arch="branch", n=1,
        emb="/home/vault/b214cb/b214cb18/regrid2/olmo/zarr/olmo_2019_N1_masked.zarr", ch=128),
    "europa_alpha_concat_N1": dict(region="europa", embed="alpha", arch="concat", n=1,
        emb="/home/vault/b214cb/b214cb18/regrid2/alpha/zarr/gcs_eu_2019_N1.zarr", ch=64),
    "europa_alpha_branch_N1": dict(region="europa", embed="alpha", arch="branch", n=1,
        emb="/home/vault/b214cb/b214cb18/regrid2/alpha/zarr/gcs_eu_2019_N1.zarr", ch=64),
    "europa_olmo_concat_N1": dict(region="europa", embed="olmo", arch="concat", n=1,
        emb="/home/vault/b214cb/b214cb18/regrid2/olmo/zarr/olmo_eu_2019_N1_masked.zarr", ch=128),
    "europa_olmo_branch_N1": dict(region="europa", embed="olmo", arch="branch", n=1,
        emb="/home/vault/b214cb/b214cb18/regrid2/olmo/zarr/olmo_eu_2019_N1_masked.zarr", ch=128),
    "taiwan_alpha_branch_N8": dict(region="cwb", embed="alpha", arch="branch", n=8,
        emb="/home/vault/b214cb/b214cb18/regrid2/alpha/zarr/gcs_2019_N8.zarr", ch=64),
    "taiwan_olmo_branch_N8": dict(region="cwb", embed="olmo", arch="branch", n=8,
        emb="/home/vault/b214cb/b214cb18/regrid2/olmo/zarr/olmo_2019_N8_masked.zarr", ch=128),
    "europa_alpha_branch_N8": dict(region="europa", embed="alpha", arch="branch", n=8,
        emb="/home/vault/b214cb/b214cb18/regrid2/alpha/zarr/gcs_eu_2019_N8.zarr", ch=64),
    "europa_olmo_branch_N8": dict(region="europa", embed="olmo", arch="branch", n=8,
        emb="/home/vault/b214cb/b214cb18/regrid2/olmo/zarr/olmo_eu_2019_N8_masked.zarr", ch=128),
}

DATA_PATH = {
    "cwb": "/data/cwa_dataset.zarr",
    "europa": "/anvme/workspace/b214cb18-ws-daniel2/wuerzburg450_corrdiff.zarr",
}
DATASET_GROUP = {"cwb": "cwb", "europa": "europa"}

REG_CKPT = "/checkpoints/exp_static2019_1_reg/{run}/v1/checkpoints_regression/UNet.0.2000128.mdlus"
RES_CKPT = "/checkpoints/exp_static2019_1_diff/{run}/v1/checkpoints_diffusion/EDMPrecondSuperResolution.0.2000128.mdlus"
OUT_NC = "/home/vault/b214cb/b214cb18/generated/exp_static2019_1_diff/{run}/{run}_{split}.nc"

# Evaluation timestamp splits: maps an output basename -> (conf file, default-group key).
# "test2021" is the canonical held-out split; "top64" is the 64 high-impact precip/wind
# events curated in timestamps_top64.csv.
SPLITS = {
    "test2021": "test_times_2021",
    "top64": "timestamps_top64",
}


def train_cfg(run: str, m: dict) -> str:
    """Diffusion training config for one run."""
    nstr = f"N{m['n']}"
    # torch_compile: ON for every diffusion run. The branch model graph-breaks and
    # falls back to eager (numerically safe), but compile still helps the others fit
    # the 24h wall — and keeping it uniform avoids the branch-N1 wall overrun seen in
    # the reg experiment.
    branch = m["arch"] == "branch"
    perf_lines = ["    torch_compile: true"]
    if m["n"] == 8:
        perf_lines.append("    dataloader_workers: 4")
    model_args = ""
    if branch:
        model_args = (
            "  model_args:\n"
            "    N_grid_channels: 4\n"
            "    embedding_type: zero\n"
            "    model_type: SongUNetEmbBranch\n"
            f"    alpha_earth_channels: {m['ch']}\n"
            "    emb_branch_channels: 64\n"
        )
    return f"""# exp_static2019_1_diff :: train :: {run}
# DIFFUSION stage, region={m['region']} embed={m['embed']} arch={m['arch']} {nstr} (static 2019 embedding).
# Mirrors conf/exp_static2019_1_reg_train_{run}.yaml but model=diffusion + the matching
# embedding-trained regression checkpoint (see diffusion_with_embeddings.md).
# NOTE: batch_size_per_gpu is pinned to 4 by exp_static2019_1_reg_common_train
# (@package _global_); bs8 gives no speedup and doubles memory (measured 2026-06-11).
hydra:
  job:
    chdir: false
    name: exp_static2019_1_diff_{run}
  run:
    dir: ./output/${{hydra:job.name}}
  searchpath:
    - pkg://conf/base
defaults:
  - dataset: {DATASET_GROUP[m['region']]}
  - model: diffusion
  - model_size: normal
  - training: ${{model}}
  - val_times_2021@_val_times
  - exp_static2019_1_reg_common_train
  - _self_
dataset:
  data_path: {DATA_PATH[m['region']]}
  embedding_path: {m['emb']}
  embedding_n: {m['n']}
model:
  embed: {m['embed']}
  embed_version: static2019_{m['arch']}_{nstr}
  hr_mean_conditioning: true
{model_args}training:
  io:
    regression_checkpoint_path: {REG_CKPT.format(run=run)}
  perf:
{chr(10).join(perf_lines)}
wandb:
  tags: [exp_static2019_1_diff, {m['region'] if m['region'] != 'cwb' else 'taiwan'}, {m['embed']}, {m['arch']}, {nstr}, static2019, diffusion]
"""


def gen_cfg(run: str, m: dict, split: str) -> str:
    """Full reg+diffusion generation config (16 ensembles) for one eval split."""
    nstr = f"N{m['n']}"
    times_conf = SPLITS[split]
    # seed_batch_size = ensemble members per forward pass (num_ensembles stays 16).
    # N8 carries a huge 3600x3600 embedding; 16-at-once OOMs even on 80 GB
    # (observed 2026-06-13, tried to alloc 49 GiB). Smaller batch -> more passes.
    seed_bs = 2 if m["n"] == 8 else 16
    return f"""# exp_static2019_1_diff :: generate :: {run} :: {split}
# Full CorrDiff inference (regression mean + diffusion residual), 16 ensembles,
# {split} split. Mirrors the Europa diffusion baseline but feeds the
# {m['embed']} embeddings and points at this run's checkpoints.
# embedding_path MUST be set — both reg and diffusion models were trained with it.
hydra:
  job:
    chdir: false
    name: gen_exp_static2019_1_diff_{run}_{split}
  run:
    dir: ./outputs/${{hydra:job.name}}
  searchpath:
    - pkg://conf/base
defaults:
  - dataset: {DATASET_GROUP[m['region']]}
  - generation: non_patched
  - {times_conf}@_eval_times
  - _self_
dataset:
  data_path: {DATA_PATH[m['region']]}
  embedding_path: {m['emb']}
  embedding_n: {m['n']}
  train: False
  all_times: True
generation:
  num_ensembles: 16
  seed_batch_size: {seed_bs}
  hr_mean_conditioning: true
  inference_mode: all
  save_input: false
  times_range: null
  times: ${{_eval_times.times}}
  io:
    reg_ckpt_filename: {REG_CKPT.format(run=run)}
    res_ckpt_filename: {RES_CKPT.format(run=run)}
    output_filename: {OUT_NC.format(run=run, split=split)}
wandb:
  mode: offline
  results_dir: "./wandb"
"""


def main():
    n = 0
    for run, m in RUNS.items():
        (CONF / f"exp_static2019_1_diff_train_{run}.yaml").write_text(train_cfg(run, m))
        n += 1
        # test2021 keeps the bare filename (backward-compatible launch name);
        # other splits get a `_<split>` suffix.
        for split in SPLITS:
            suffix = "" if split == "test2021" else f"_{split}"
            (CONF / f"exp_static2019_1_diff_gen_{run}{suffix}.yaml").write_text(
                gen_cfg(run, m, split)
            )
            n += 1
        print(f"✓ {run}")
    print(f"✓ wrote {n} configs to {CONF}")


if __name__ == "__main__":
    main()
