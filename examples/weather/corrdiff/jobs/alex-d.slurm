#!/bin/bash -l
#SBATCH -J CorrDiff
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --partition=a100
#SBATCH --output=output/alex/jobout/job-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user "daniel.wilhelm@stud-mail.uni-wuerzburg.de"
#SBATCH --constraint=a100_80

# --- GPU config: set to 1 or 4 ---
NUM_GPUS=4

set -x  # Zeigt jeden Befehl an, bevor er ausgeführt wird

# Set up environment
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80



export WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
echo ${CUDA_VISIBLE_DEVICES}


source $HOME/.job_secrets

DATASET_SRC=/anvme/workspace/b214cb18-ws-daniel-cwb/cwa_dataset
DATASET_TAR=/anvme/workspace/b214cb18-ws-daniel-cwb/cwa_dataset.tar
COPY_TO_TMPDIR=false  # set to false to use dataset directly from source


if [ "$COPY_TO_TMPDIR" = true ]; then
  echo "Extracting dataset to $TMPDIR..."
  START=$(date +%s)

  tar xf "$DATASET_TAR" -C "$TMPDIR"
  END=$(date +%s)
  echo "Extraction done in $((END - START)) seconds."
  DATA_DIR=$TMPDIR/cwa_dataset
else
  echo "Using dataset directly from $DATASET_SRC"
  DATA_DIR=$DATASET_SRC
fi

export HYDRA_FULL_ERROR=1

apptainer exec $HOME/apptainer/corrdiff_10_02.sif python3 -c "import zarr; print('zarr version:', zarr.__version__)"

#try without srun, dann nicht doppelt, srun hier nur notwendig wenn mehrere noes
apptainer run --nv --bind $DATA_DIR:/data/ \
                        --bind $WORK/checkpoints:/checkpoints/ \
                        $HOME/apptainer/corrdiff_10_02.sif torchrun --nproc_per_node=$NUM_GPUS  $HOME/corrdiff/train.py --config-name config_training_taiwan_regression-alex


#srun apptainer run --nv --bind /anvme/workspace/b214cb13-ecodata/downscaling/:/data/ --bind $WORK/checkpoints:/checkpoints/ $WORK/own_corrdiff.sif python -u $HOME/corrdiff/generate.py --config-name