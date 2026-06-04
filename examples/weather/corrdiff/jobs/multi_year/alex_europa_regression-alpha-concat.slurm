#!/bin/bash -l
#SBATCH -J reg_eu_alpha_concat
#SBATCH --ntasks=1
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --partition=a100
#SBATCH --output=output/alex/jobout/job-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user "daniel.wilhelm@stud-mail.uni-wuerzburg.de"
#SBATCH --constraint=el9   # any a100 (concat adds only 64 input chans; 40 GB suffices)

# Europa regression with YEAR-MATCHED Alpha Earth embeddings — PLAIN CONCAT
# ("alpha-concat"): the 64 alpha channels are appended directly to the ERA5
# input of the regression UNet (no separate branch).

NUM_GPUS=4

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

export WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
echo ${CUDA_VISIBLE_DEVICES}

source $HOME/.job_secrets

# Europa training store (zarr v3) lives on /anvme; bound 1:1.
EUROPA_WS=/anvme/workspace/b214cb18-ws-daniel2

# Per-year embedding stores live under $HPCVAULT/regrid2 (= /home/vault/...).
# /home/vault is NOT auto-mounted on Alex, so bind it explicitly.

mkdir -p $WORK/checkpoints/europa

export HYDRA_FULL_ERROR=1

# Container: zarr v3 image required (Europa store + embedding stores are zarr v3).
CONTAINER=$HOME/apptainer/corrdiff_zarr3.sif

apptainer exec $CONTAINER python3 -c "import zarr; print('zarr version:', zarr.__version__)"

MASTER_PORT=$((20000 + SLURM_JOB_ID % 30000))

apptainer run --nv \
    --bind ${EUROPA_WS}:${EUROPA_WS} \
    --bind /home/vault:/home/vault \
    --bind $WORK/checkpoints:/checkpoints/ \
    $CONTAINER \
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
        $HOME/corrdiff/train.py \
        --config-name multi_year/config_training_europa_regression-alpha-concat
