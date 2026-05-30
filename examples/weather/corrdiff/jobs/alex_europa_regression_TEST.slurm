#!/bin/bash -l
#SBATCH -J reg_eu_TEST_40gb
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=a100
#SBATCH --output=output/alex/jobout/job-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user "daniel.wilhelm@stud-mail.uni-wuerzburg.de"
#SBATCH --constraint=a100_40

# Short verification job: confirms torch_compile + find_unused=False + new
# wandb logging do not break the training loop. Writes to a separate
# checkpoint dir (reg_eu_TEST_compile) so the real run's checkpoints are
# untouched. Delete this file once the test passes.

# --- GPU config: set to 1, 2 or 4 ---
NUM_GPUS=2

# Set up environment
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

export WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
echo ${CUDA_VISIBLE_DEVICES}

source $HOME/.job_secrets

EUROPA_WS=/anvme/workspace/b214cb18-ws-daniel2

mkdir -p $WORK/checkpoints/europa

export HYDRA_FULL_ERROR=1

CONTAINER=$HOME/apptainer/corrdiff_zarr3.sif

apptainer exec $CONTAINER python3 -c "import zarr; print('zarr version:', zarr.__version__)"

# Unique torchrun rendezvous port per SLURM job to avoid EADDRINUSE
# when another job is also running torchrun on the same node.
MASTER_PORT=$((20000 + SLURM_JOB_ID % 30000))

apptainer run --nv \
    --bind ${EUROPA_WS}:${EUROPA_WS} \
    --bind $WORK/checkpoints:/checkpoints/ \
    $CONTAINER \
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
        $HOME/corrdiff/train.py \
        --config-name config_training_europa_regression-alex-TEST
