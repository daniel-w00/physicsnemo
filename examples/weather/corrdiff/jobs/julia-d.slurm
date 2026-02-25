#!/bin/bash
#SBATCH -J PrecipitationDownscaling
#SBATCH -c 16
#SBATCH -p standard
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daniel.wilhelm@stud-mail.uni-wuerzburg.de
#SBATCH --output=output/job-%j.out
#SBATCH --qos=prof-x

NUM_GPUS=2


source ~/.job_secrets

export WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
export APPTAINERENV_WANDB_API_KEY=${WANDB_API_KEY}
echo ${CUDA_VISIBLE_DEVICES}


#srun apptainer run --nv --bind ~/datatemp/CorrDiff:/data ~/corrdiff.sif python -u ~/bjerknes/corrdiff/train.py --config-name $1 \

srun apptainer run --nv --bind ~/data/hrrr_mini:/data --bind ~/data/checkpoints:/checkpoints/ ~/apptainer/new_corrdiff.sif python -u ~/bjerknes/train.py --config-name $1 \

DATASET_SRC=/data/42-julia-hpc-rz-lsx/sih25nq/downscaling/CorrDiff/cwa_dataset
DATASET_TAR=/data/42-julia-hpc-rz-lsx/s373395/datasets/cwa_dataset.tar
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


# julia old command
apptainer run --nv --bind /tmp/:/data --bind $HOME/checkpoints:/checkpoints/   \
   $HOME/apptainer/corrdiff_10_02.sif torchrun torchrun --nproc_per_node=$NUM_GPUS   \
   ~/bjerknes/corrdiff/train.py --config-nameconfig_training_taiwan_diffusion-alex