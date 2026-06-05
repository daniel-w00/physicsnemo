#!/bin/bash -l
#SBATCH -J exp_static2019_1_reg_tr
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --partition=a100
#SBATCH --output=output/alex/jobout/job-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user "daniel.wilhelm@stud-mail.uni-wuerzburg.de"
# NOTE: --constraint (a100_80 for N8) is added on the CLI by submit_experiment.sh.

# Generic training job for exp_static2019_1_reg. Driven by env vars exported by
# submit_experiment.sh:
#   REGION     cwb | europa     (selects dataset bind)
#   TRAIN_CFG  e.g. train_taiwan_olmo_branch_N8  (under conf/exp_static2019_1_reg/)
#   CKPT_DIR   in-container checkpoint dir, e.g. /checkpoints/exp_static2019_1_reg/<run>/<version>
#   NUM_GPUS   default 4
: "${NUM_GPUS:=4}"
: "${EXTRA_ARGS:=}"

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
export HYDRA_FULL_ERROR=1
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"

source $HOME/.job_secrets

# Container: zarr v3 image (embedding stores — incl. N8 — are zarr v3).
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
        --config-name exp_static2019_1_reg_$TRAIN_CFG \
        ++training.io.checkpoint_dir=$CKPT_DIR \
        $EXTRA_ARGS
# EXTRA_ARGS (optional, exported by caller): extra Hydra overrides appended verbatim,
# e.g. EXTRA_ARGS='training.hp.training_duration=2000 training.hp.batch_size_per_gpu=2'
