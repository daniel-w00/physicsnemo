#!/bin/bash
#SBATCH -J PrecipitationDownscaling
#SBATCH -c 16
#SBATCH -p standard
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daniel.wilhelm@stud-mail.uni-wuerzburg.de
#SBATCH --output=output/job-%j.out

source ~/.job_secrets

export WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
export APPTAINERENV_WANDB_API_KEY=${WANDB_API_KEY}
echo ${CUDA_VISIBLE_DEVICES}


#srun apptainer run --nv --bind ~/datatemp/CorrDiff:/data ~/corrdiff.sif python -u ~/bjerknes/corrdiff/train.py --config-name $1 \

srun apptainer run --nv --bind ~/data/hrrr_mini:/data --bind ~/data/checkpoints:/checkpoints/ ~/apptainer/new_corrdiff.sif python -u ~/bjerknes/train.py --config-name $1 \

