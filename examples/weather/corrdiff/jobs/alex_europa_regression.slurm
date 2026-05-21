#!/bin/bash -l
#SBATCH -J reg_eu_pure_v1
#SBATCH --ntasks=1
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=a100
#SBATCH --output=output/alex/jobout/job-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user "daniel.wilhelm@stud-mail.uni-wuerzburg.de"
#SBATCH --constraint=a100_80

# --- GPU config: set to 1, 2 or 4 ---
NUM_GPUS=2

# Set up environment
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

export WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
echo ${CUDA_VISIBLE_DEVICES}

source $HOME/.job_secrets

# Europa training store (zarr v3) lives on /anvme; bound 1:1 so the
# absolute path inside the container matches the path in the config.
EUROPA_WS=/anvme/workspace/b214cb18-ws-daniel2

# Europa-dedicated checkpoint root (separate from taiwan/).
# Container sees this as /checkpoints/europa/reg_eu_pure_v1/...
mkdir -p $WORK/checkpoints/europa

export HYDRA_FULL_ERROR=1

# Container: zarr v3 image required — Europa store is zarr v3.
CONTAINER=$HOME/apptainer/corrdiff_zarr3.sif

apptainer exec $CONTAINER python3 -c "import zarr; print('zarr version:', zarr.__version__)"

apptainer run --nv \
    --bind ${EUROPA_WS}:${EUROPA_WS} \
    --bind $WORK/checkpoints:/checkpoints/ \
    $CONTAINER \
    torchrun --nproc_per_node=$NUM_GPUS \
        $HOME/corrdiff/train.py \
        --config-name config_training_europa_regression-alex
