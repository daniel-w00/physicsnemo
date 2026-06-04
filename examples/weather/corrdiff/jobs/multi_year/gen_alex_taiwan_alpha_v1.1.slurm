#!/bin/bash -l
#SBATCH -J gen_taiwan_alpha_v1.1
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --output=output/alex/jobout/gen-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user "daniel.wilhelm@stud-mail.uni-wuerzburg.de"
#SBATCH --constraint=el9

# Generation: plain-concat YEAR-MATCHED Alpha regression (alpha-v1.1), regression
# only, on the held-out 2021 test split. Single GPU on purpose (avoids multi-rank
# NetCDF-writer coordination). See alpha-integrate.md.

# Set up environment
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

export WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
echo ${CUDA_VISIBLE_DEVICES}

source $HOME/.job_secrets

# Taiwan CWA store (zarr v2) — current Alex location (ws-daniel-cwb expired).
DATASET_SRC=/anvme/workspace/b214cb18-ws-daniel3/b214cb18-ws-daniel-cwb-1779073861/cwa_dataset

# Output dir on HPCVAULT (config writes the NetCDF here). Vault is not auto-mounted.
OUT_DIR=$HPCVAULT/generated
mkdir -p $OUT_DIR

export HYDRA_FULL_ERROR=1

# Container: zarr v3 image required (embedding stores are zarr v3).
CONTAINER=$HOME/apptainer/corrdiff_zarr3.sif

apptainer exec $CONTAINER python3 -c "import zarr; print('zarr version:', zarr.__version__)"

# Launch under torchrun (single process) so RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT
# are set (physicsnemo's DistributedManager needs them). Bare python3 leaves them unset.
MASTER_PORT=$((20000 + SLURM_JOB_ID % 30000))

apptainer run --nv \
    --bind $DATASET_SRC:/data/ \
    --bind /home/vault:/home/vault \
    --bind $HPCVAULT/checkpoints:/checkpoints/ \
    $CONTAINER \
    torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
        $HOME/corrdiff/generate.py \
        --config-name multi_year/config_generate_taiwan-alpha-v1.1
