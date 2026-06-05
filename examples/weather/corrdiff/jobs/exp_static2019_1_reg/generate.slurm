#!/bin/bash -l
#SBATCH -J exp_static2019_1_reg_gen
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --output=output/alex/jobout/gen-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user "daniel.wilhelm@stud-mail.uni-wuerzburg.de"
# NOTE: --constraint (a100_80 for N8) is added on the CLI by submit_experiment.sh.

# Generic generation job for exp_static2019_1_reg. Single GPU (NetCDF writer).
# Driven by env vars exported by submit_experiment.sh:
#   REGION      cwb | europa
#   GEN_CFG     e.g. gen_taiwan_olmo_branch_N8  (under conf/exp_static2019_1_reg/)
#   CKPT_DIR    in-container checkpoint dir of the matching train run
#   OUT_FILE    in-container output NetCDF path, e.g. /generated/.../test2021.nc
#   TIMES_MODE  test | year   (year overrides generation.times to the full-year set)

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
export HYDRA_FULL_ERROR=1
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"

source $HOME/.job_secrets

CONTAINER=$HOME/apptainer/corrdiff_zarr3.sif

# Final regression checkpoint is deterministic: total_batch_size=256,
# training_duration=2_000_000 => UNet.0.2000128.mdlus.
CKPT_HOST="$WORK/checkpoints${CKPT_DIR#/checkpoints}"
CKPT_FILE="$CKPT_HOST/checkpoints_regression/UNet.0.2000128.mdlus"
if [ ! -f "$CKPT_FILE" ]; then
    echo "⚠️  training not finished: $CKPT_FILE missing — skipping generation." >&2
    exit 1
fi

# $WORK/generated is bound to /generated; create the host-side output dir.
OUT_FILE_HOST="$WORK/generated${OUT_FILE#/generated}"
mkdir -p "$(dirname "$OUT_FILE_HOST")"

# Year mode swaps the default test split for the committed full-year sample.
TIMES_OVERRIDE=""
if [ "$TIMES_MODE" = "year" ]; then
    TIMES_OVERRIDE='generation.times=${_year_times.times}'
fi

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
    --bind $WORK/generated:/generated/ \
    $CONTAINER \
    torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
        $HOME/corrdiff/generate.py \
        --config-name exp_static2019_1_reg_$GEN_CFG \
        ++generation.io.reg_ckpt_filename=$CKPT_DIR/checkpoints_regression/UNet.0.2000128.mdlus \
        ++generation.io.output_filename=$OUT_FILE $TIMES_OVERRIDE
