#!/bin/bash
#SBATCH -J PrecipitationDownscaling
#SBATCH -c 16
#SBATCH -p test
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.wilhelm@stud-mail.uni-wuerzburg.de
#SBATCH --output=output/julia2/jobout/job-%j.out


export WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
echo ${CUDA_VISIBLE_DEVICES}

export HYDRA_FULL_ERROR=1

srun apptainer run --nv --bind ~/data/hrrr_mini:/data \
                        --bind ~/data/checkpoints/hrrr_mini:/checkpoints/ \
                        ~/apptainer/new_corrdiff.sif python -u ~/bjerknes/generate.py --config-name $1 \
