#!/bin/bash -l
#SBATCH -J reg_eu_pure_v1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --output=output/job-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user "daniel.wilhelm@stud-mail.uni-wuerzburg.de"

# Environment
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
export HYDRA_FULL_ERROR=1
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Container: zarr v3 image required — Europa store is zarr v3.
CONTAINER=$HOME/apptainer/corrdiff_zarr3.sif

# Data on /anvme, bound 1:1 so the absolute path inside == outside.
EUROPA_WS=/anvme/workspace/b214cb18-ws-daniel2

# Europa-dedicated checkpoint root (separate from taiwan/).
# Container sees this as /checkpoints/europa/reg_eu_pure_v1/...
mkdir -p $WORK/checkpoints/europa

srun apptainer run --nv \
    --bind ${EUROPA_WS}:${EUROPA_WS} \
    --bind $WORK/checkpoints:/checkpoints/ \
    ${CONTAINER} \
    python -u $HOME/corrdiff/train.py \
        --config-name config_training_europa_regression-alex
