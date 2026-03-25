#!/bin/bash -l
#SBATCH -J Gen-CorrDiff
#SBATCH --ntasks=1
#SBATCH --time=1:59:00
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --output=output/alex/jobout/job-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user "daniel.wilhelm@stud-mail.uni-wuerzburg.de"

# Set up environment
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

export WANDB__SERVICE_WAIT=300


export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
echo ${CUDA_VISIBLE_DEVICES}

export HYDRA_FULL_ERROR=1

#srun apptainer run --nv --bind ~/data/hrrr_mini:/data \
 #                       --bind ~/data/checkpoints/hrrr_mini:/checkpoints/ \
  #                      ~/apptainer/new_corrdiff.sif python -u ~/corrdiff/generate.py  --config-name config_generate_taiwan


srun apptainer run --nv --bind /anvme/workspace/b214cb18-ws-daniel-cwb/cwa_dataset:/data/ --bind $HPCVAULT/checkpoints:/checkpoints/ --bind $HOME/results:/results $HOME/apptainer/corrdiff_10_02.sif python -u $HOME/corrdiff/generate.py  --config-name config_generate_taiwan