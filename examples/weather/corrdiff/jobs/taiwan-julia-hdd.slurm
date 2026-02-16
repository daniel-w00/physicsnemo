#!/bin/bash
#SBATCH -J taiwan-corrdiff-hdd
#SBATCH -c 16
#SBATCH -p standard
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.wilhelm@stud-mail.uni-wuerzburg.de
#SBATCH --output=output/job-%j.out


source ~/.job_secrets

export WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
export APPTAINERENV_WANDB_API_KEY=${WANDB_API_KEY}
echo "CUDA Devices: ${CUDA_VISIBLE_DEVICES}"



# 5. Training starten
echo "Starte Training im Apptainer-Container..."


srun apptainer run --nv \
    --bind /data/42-julia-hpc-rz-lsx/sih25nq/downscaling/CorrDiff/cwa_dataset:/data \
    --bind ~/data/checkpoints/taiwan:/checkpoints/ \
    ~/apptainer/new_corrdiff.sif \
    python -u ~/bjerknes/train.py --config-name $1

echo "srun-Befehl beendet."

