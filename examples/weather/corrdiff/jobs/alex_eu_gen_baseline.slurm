#!/bin/bash -l
#SBATCH -J gen_eu_base
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --output=output/alex/jobout/gen-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user "daniel.wilhelm@stud-mail.uni-wuerzburg.de"
#SBATCH --constraint=a100_80   # headroom; generation is short so queue wait is minor

# Regression-only Europa baseline (reg_eu_pure_v1, step 2000128). Single GPU on
# purpose: avoids multi-rank NetCDF-writer coordination.
#
# Split is selected by the TIMES_MODE env var (default: test):
#   sbatch --export=ALL,TIMES_MODE=test jobs/alex_eu_gen_baseline.slurm  -> ...-test2021.nc
#   sbatch --export=ALL,TIMES_MODE=year jobs/alex_eu_gen_baseline.slurm  -> ...-year2021.nc

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
export HYDRA_FULL_ERROR=1
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"

source $HOME/.job_secrets

CONTAINER=$HOME/apptainer/corrdiff_zarr3.sif   # zarr v3 image — Europa store is zarr v3
EUROPA_WS=/anvme/workspace/b214cb18-ws-daniel2

# Baseline run + checkpoint step (final regression checkpoint, total_batch_size=256,
# training_duration=2_000_000 => step 2000128). The .mdlus is the sibling of
# checkpoint.0.<step>.pt. Override RUN_NAME/CKPT_STEP via --export for other runs.
RUN_NAME=${RUN_NAME:-reg_eu_pure_v1}
CKPT_STEP=${CKPT_STEP:-2000128}
CKPT_IN=/checkpoints/europa/$RUN_NAME/checkpoints_regression/UNet.0.$CKPT_STEP.mdlus
CKPT_FILE="$WORK/checkpoints/europa/$RUN_NAME/checkpoints_regression/UNet.0.$CKPT_STEP.mdlus"
if [ ! -f "$CKPT_FILE" ]; then
    echo "❌ regression checkpoint missing: $CKPT_FILE" >&2
    exit 1
fi

# Structured output: <vault>/generated/europa_baseline/<run>/<run>-step<step>-<split>.nc
# The checkpoint step is embedded in the filename so files stay self-describing.
OUT_DIR=$HPCVAULT/generated/europa_baseline/$RUN_NAME

# Pick the split + output path.
TIMES_MODE=${TIMES_MODE:-test}
case "$TIMES_MODE" in
  test)
    OUT_FILE=$OUT_DIR/${RUN_NAME}-step${CKPT_STEP}-test2021.nc
    TIMES_OVERRIDE=""
    ;;
  year)
    OUT_FILE=$OUT_DIR/${RUN_NAME}-step${CKPT_STEP}-year2021.nc
    TIMES_OVERRIDE='generation.times=${_year_times.times}'
    ;;
  *)
    echo "❌ unknown TIMES_MODE=$TIMES_MODE (use test|year)" >&2; exit 1 ;;
esac
mkdir -p "$OUT_DIR"
echo "RUN_NAME=$RUN_NAME CKPT_STEP=$CKPT_STEP TIMES_MODE=$TIMES_MODE -> $OUT_FILE"

apptainer exec $CONTAINER python3 -c "import zarr; print('zarr version:', zarr.__version__)"

# torchrun (single process) so RANK/WORLD_SIZE/MASTER_* are set — matches the
# training/generation jobs. /home/vault is NOT auto-mounted, so bind it explicitly.
MASTER_PORT=$((20000 + SLURM_JOB_ID % 30000))

apptainer run --nv \
    --bind ${EUROPA_WS}:${EUROPA_WS} \
    --bind /home/vault:/home/vault \
    --bind $WORK/checkpoints:/checkpoints/ \
    $CONTAINER \
    torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
        $HOME/corrdiff/generate.py \
        --config-name config_generate_europa_baseline-alex \
        ++generation.io.reg_ckpt_filename=$CKPT_IN \
        ++generation.io.output_filename=$OUT_FILE $TIMES_OVERRIDE
