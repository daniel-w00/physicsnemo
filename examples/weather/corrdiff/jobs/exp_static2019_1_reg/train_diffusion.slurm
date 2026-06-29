#!/bin/bash -l
#SBATCH -J exp_static2019_1_diff_tr
#SBATCH --ntasks=1
#SBATCH --time=17:00:00
#SBATCH --gres=gpu:a100:8
#SBATCH --partition=a100
#SBATCH --output=output/alex/jobout/job-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user "daniel.wilhelm@stud-mail.uni-wuerzburg.de"
# NOTE: --constraint (e.g. a100_80) can be added on the CLI by the caller.

# Generic DIFFUSION training job for exp_static2019_1_diff. Driven by env vars:
#   REGION       cwb | europa            (selects dataset bind)
#   CONFIG_NAME  full Hydra config name, e.g. exp_static2019_1_diff_train_europa_alpha_concat_N1
#   CKPT_DIR     in-container checkpoint dir, e.g. /checkpoints/exp_static2019_1_diff/<run>/<version>
#   NUM_GPUS     default 8
: "${NUM_GPUS:=8}"
: "${EXTRA_ARGS:=}"

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
export HYDRA_FULL_ERROR=1
echo "GPUs: ${CUDA_VISIBLE_DEVICES}  CONFIG_NAME=${CONFIG_NAME}"

source $HOME/.job_secrets

# Container: zarr v3 image (embedding stores are zarr v3).
CONTAINER=$HOME/apptainer/corrdiff_zarr3.sif

# $WORK/checkpoints is bound to /checkpoints; CKPT_DIR is the in-container path.
CKPT_HOST="$WORK/checkpoints${CKPT_DIR#/checkpoints}"
mkdir -p "$CKPT_HOST"

# Region-specific dataset bind. Both regions need /home/vault for the embeddings.
case "$REGION" in
  cwb)
    DATASET_SRC=/anvme/workspace/b214cb18-ws-daniel3/b214cb18-ws-daniel-cwb-1779073861/cwa_dataset
    DATA_BIND="--bind $DATASET_SRC:/data/"
    ;;
  europa)
    EUROPA_WS=/anvme/workspace/b214cb18-ws-daniel2
    DATA_BIND="--bind $EUROPA_WS:$EUROPA_WS"
    ;;
  *)
    echo "❌ unknown REGION=$REGION" >&2; exit 1 ;;
esac

apptainer exec $CONTAINER python3 -c "import zarr; print('zarr version:', zarr.__version__)"

MASTER_PORT=$((20000 + SLURM_JOB_ID % 30000))

apptainer run --nv \
    $DATA_BIND \
    --bind /home/vault:/home/vault \
    --bind $WORK/checkpoints:/checkpoints/ \
    $CONTAINER \
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
        $HOME/corrdiff/train.py \
        --config-name $CONFIG_NAME \
        ++training.io.checkpoint_dir=$CKPT_DIR \
        $EXTRA_ARGS
