#!/bin/bash -l
#SBATCH -J CorrDiff
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --partition=a100
#SBATCH --output=output/job-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user "daniel.wilhelm@stud-mail.uni-wuerzburg.de"
#SBATCH --constraint=a100_80

set -x  # Zeigt jeden Befehl an, bevor er ausgeführt wird

# Set up environment
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80



export WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
echo ${CUDA_VISIBLE_DEVICES}


source $HOME/.job_secrets

#/anvme/workspace/b214cb18-ws-daniel-cwb/

# Copy dataset to fast local storage
 echo "Copying dataset to $TMPDIR..."
 START=$(date +%s)
 cp -r /anvme/workspace/b214cb18-ws-daniel-cwb/cwa_dataset $TMPDIR/
 END=$(date +%s)
 echo "Copy done in $((END - START)) seconds."

export HYDRA_FULL_ERROR=1

#srun apptainer run --nv --bind /anvme/workspace/b214cb18-ws-daniel-cwb/cwa_dataset:/data/ \
  #                      --bind $WORK/checkpoints:/checkpoints/ \
    #                    $HOME/apptainer/corrdiff_10_02.sif python -u  $HOME/corrdiff/train.py --config-name config_training_taiwan_regression-alex

srun apptainer run --nv --bind $TMPDIR/cwa_dataset/:/data/ \
                        --bind $WORK/checkpoints:/checkpoints/ \
                        $HOME/apptainer/corrdiff_10_02.sif torchrun --nproc_per_node=4  $HOME/corrdiff/train.py --config-name config_training_taiwan_regression-alex


#srun apptainer run --nv --bind /anvme/workspace/b214cb13-ecodata/downscaling/:/data/ --bind $WORK/checkpoints:/checkpoints/ $WORK/own_corrdiff.sif python -u $HOME/bjerknes/corrdiff/generate.py --config-name 