#!/bin/bash -l
#SBATCH -J taiwan_N8_TEST
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --output=output/alex/jobout/job-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user "daniel.wilhelm@stud-mail.uni-wuerzburg.de"
#SBATCH --constraint=a100_80&el9   # 80 GB A100: N8 is ~6.6 GB/sample at the branch input

# SMOKE TEST: OLMO N8 separate-tensor emb-branch path runs end-to-end on real
# data for a handful of samples (+ one validation). 1 GPU, bs=1, dataloader
# workers=0 (set in the TEST config so the resident N8 field is not duplicated).

NUM_GPUS=1

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

export WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
echo ${CUDA_VISIBLE_DEVICES}

source $HOME/.job_secrets

# Taiwan CWA store (zarr v2) — current Alex location.
DATASET_SRC=/anvme/workspace/b214cb18-ws-daniel3/b214cb18-ws-daniel-cwb-1779073861/cwa_dataset

# Per-year embedding stores under $HPCVAULT/regrid2; /home/vault must be bound.
mkdir -p $HPCVAULT/checkpoints/taiwan

export HYDRA_FULL_ERROR=1

# Container: zarr v3 image required (embedding stores — incl. N8 — are zarr v3).
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
        --config-name config_training_taiwan_regression-emb_branch-year-N8-TEST
