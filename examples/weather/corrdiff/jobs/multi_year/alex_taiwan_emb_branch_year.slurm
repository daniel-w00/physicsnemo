#!/bin/bash -l
#SBATCH -J taiwan_emb_branch_year
#SBATCH --ntasks=1
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --partition=a100
#SBATCH --output=output/alex/jobout/job-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user "daniel.wilhelm@stud-mail.uni-wuerzburg.de"
#SBATCH --constraint=a100_80&el9   # 80 GB A100 (branch fusion OOMs on 40 GB at bs=4)

# Taiwan regression with YEAR-MATCHED Alpha Earth embeddings — PARALLEL CNN
# ("emb_branch_year_1"): the 64 alpha channels are split off and processed by a
# dedicated CNN (SongUNetEmbBranch), fused before the first downsampling.
# See alpha-integrate.md.

# --- GPU config: set to 1, 2 or 4 ---
NUM_GPUS=4

# Set up environment
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

export WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
echo ${CUDA_VISIBLE_DEVICES}

source $HOME/.job_secrets

# Taiwan CWA store (zarr v2) — current Alex location (ws-daniel-cwb expired).
DATASET_SRC=/anvme/workspace/b214cb18-ws-daniel3/b214cb18-ws-daniel-cwb-1779073861/cwa_dataset

# Per-year embedding stores under $HPCVAULT/regrid2; /home/vault must be bound.
mkdir -p $HPCVAULT/checkpoints/taiwan

export HYDRA_FULL_ERROR=1

# Container: zarr v3 image required (embedding stores are zarr v3).
CONTAINER=$HOME/apptainer/corrdiff_zarr3.sif

apptainer exec $CONTAINER python3 -c "import zarr; print('zarr version:', zarr.__version__)"

MASTER_PORT=$((20000 + SLURM_JOB_ID % 30000))

apptainer run --nv \
    --bind $DATASET_SRC:/data/ \
    --bind /home/vault:/home/vault \
    --bind $HPCVAULT/checkpoints:/checkpoints/ \
    $CONTAINER \
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
        $HOME/corrdiff/train.py \
        --config-name multi_year/config_training_taiwan_regression-emb_branch-year
