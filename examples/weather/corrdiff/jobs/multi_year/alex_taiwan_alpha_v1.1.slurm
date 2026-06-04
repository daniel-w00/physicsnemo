#!/bin/bash -l
#SBATCH -J taiwan_alpha_v1.1
#SBATCH --ntasks=1
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --partition=a100
#SBATCH --output=output/alex/jobout/job-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user "daniel.wilhelm@stud-mail.uni-wuerzburg.de"
#SBATCH --constraint=el9   # any a100 (40 or 80 GB); bs=4 needs ~24 GB

# Taiwan regression with YEAR-MATCHED Alpha Earth embeddings — PLAIN CONCAT
# ("alpha-v1.1"): the 64 alpha channels are appended directly to the ERA5 input
# of the regression UNet. See alpha-integrate.md.

# --- GPU config: set to 1, 2 or 4 ---
NUM_GPUS=4

# Set up environment
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

export WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
echo ${CUDA_VISIBLE_DEVICES}

source $HOME/.job_secrets

# Taiwan CWA store (zarr v2) — current Alex location. The old
# ws-daniel-cwb workspace expired; the live copy sits in ws-daniel3.
# Bound 1:1 to /data so the config path /data/cwa_dataset.zarr resolves.
DATASET_SRC=/anvme/workspace/b214cb18-ws-daniel3/b214cb18-ws-daniel-cwb-1779073861/cwa_dataset

# Per-year embedding stores live under $HPCVAULT/regrid2 (= /home/vault/...).
# /home/vault is NOT auto-mounted on Alex, so bind it explicitly; the config's
# embedding_root (/home/vault/b214cb/b214cb18/regrid2) must resolve inside.
mkdir -p $HPCVAULT/checkpoints/taiwan

export HYDRA_FULL_ERROR=1

# Container: zarr v3 image required — the embedding stores are zarr v3.
# (It also reads the zarr-v2 CWA store fine via auto-detection.)
CONTAINER=$HOME/apptainer/corrdiff_zarr3.sif

apptainer exec $CONTAINER python3 -c "import zarr; print('zarr version:', zarr.__version__)"

# Unique torchrun rendezvous port per job (Alex shares nodes between jobs).
MASTER_PORT=$((20000 + SLURM_JOB_ID % 30000))

apptainer run --nv \
    --bind $DATASET_SRC:/data/ \
    --bind /home/vault:/home/vault \
    --bind $HPCVAULT/checkpoints:/checkpoints/ \
    $CONTAINER \
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
        $HOME/corrdiff/train.py \
        --config-name multi_year/config_training_taiwan_regression-alpha-v1.1
