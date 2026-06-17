#!/bin/bash -l
#SBATCH -J gen_eu_v1
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --output=output/alex/jobout/gen-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user "daniel.wilhelm@stud-mail.uni-wuerzburg.de"
#SBATCH --constraint=a100_80   # headroom for seed_batch_size; generation is short so queue wait is minor

# Single GPU on purpose: avoids multi-rank NetCDF-writer coordination.
# Generation throughput comes from seed_batch_size (ensemble members per forward pass).

# Set up environment
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

export WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
echo ${CUDA_VISIBLE_DEVICES}

source $HOME/.job_secrets

# Dataset bind is region-aware (REGION env; defaults to europa for backward compat).
#   europa -> Würzburg store bound 1:1 (absolute path matches the config)
#   cwb    -> Taiwan CWA store bound to /data (config uses /data/cwa_dataset.zarr)
case "${REGION:-europa}" in
  europa)
    EUROPA_WS=/anvme/workspace/b214cb18-ws-daniel2
    DATA_BIND="--bind ${EUROPA_WS}:${EUROPA_WS}"
    ;;
  cwb)
    DATASET_SRC=/anvme/workspace/b214cb18-ws-daniel3/b214cb18-ws-daniel-cwb-1779073861/cwa_dataset
    DATA_BIND="--bind ${DATASET_SRC}:/data/"
    ;;
  *)
    echo "❌ unknown REGION=${REGION}" >&2; exit 1 ;;
esac

# Output directory on HPCVAULT (config writes the NetCDF here). Vault is NOT auto-mounted
# in apptainer, so it must be bound explicitly below.
OUT_DIR=$HPCVAULT/generated
mkdir -p $OUT_DIR

export HYDRA_FULL_ERROR=1

# Container: zarr v3 image required — Europa store is zarr v3.
CONTAINER=$HOME/apptainer/corrdiff_zarr3.sif

apptainer exec $CONTAINER python3 -c "import zarr; print('zarr version:', zarr.__version__)"

# Launch under torchrun (single process) so RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT
# are set. Bare `python3` leaves them unset, so physicsnemo's DistributedManager
# falls through to the SLURM path, which needs SLURM_LAUNCH_NODE_IPADDR — only set
# inside an srun step, not a bare apptainer run. Matches the training jobs.
MASTER_PORT=$((20000 + SLURM_JOB_ID % 30000))

apptainer run --nv \
    $DATA_BIND \
    --bind $WORK/checkpoints:/checkpoints/ \
    --bind $HPCVAULT/generated:$HPCVAULT/generated \
    --bind $HPCVAULT/regrid2:$HPCVAULT/regrid2 \
    $CONTAINER \
    torchrun --nproc_per_node=${NUM_GPUS:-1} --master_port=$MASTER_PORT \
        $HOME/corrdiff/generate.py \
        --config-name ${CONFIG_NAME:-config_generate_europa-alex}
