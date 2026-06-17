#!/bin/bash -l
#SBATCH -J exp_static2019_1_reg_eval
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --output=output/alex/jobout/eval-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user "daniel.wilhelm@stud-mail.uni-wuerzburg.de"

# Offline eval_pipeline compares for exp_static2019_1_reg.
# CPU-only work (torch CPU backend); a GPU is allocated only because Alex has no
# CPU-only partition — it is not used. Runs all 4 compares sequentially:
#   taiwan/test, taiwan/year, europa/test, europa/year   (6 runs overlaid each).
# A run is SKIPPED if its metrics_comparison.csv already exists (resumable).
#
# NOTE: europa/* currently FAIL on the missing `maximum_radar_reflectivity`
# variable — the eval_pipeline hardcodes Taiwan's channel names and Europa stores
# `precipitation_amount_1hr` instead. They need the io.py rename change to run.
# They are kept here (non-fatal) so taiwan completes regardless.

set -u
cd "$HOME/corrdiff"

CONTAINER=$HOME/apptainer/corrdiff_zarr3.sif
GEN=$WORK/generated/exp_static2019_1_reg
RUNS="alpha_concat_N1 alpha_branch_N1 olmo_concat_N1 olmo_branch_N1 alpha_branch_N8 olmo_branch_N8"

run_compare () {  # $1=region (taiwan|europa)  $2=split (test|year)
    local region=$1 split=$2 args="" extra=""
    local od="analysis/eval_pipeline/results/${region}_${split}"

    if ls "$od"/*/metrics_comparison.csv >/dev/null 2>&1; then
        echo "⏭️  skip ${region}/${split} (results already exist in $od)"
        return 0
    fi

    for r in $RUNS; do
        args="$args --pred $GEN/${region}_${r}/v1/${split}2021.nc --name ${r}"
    done
    # Default sample panels (n_samples=3). Do NOT pass --times-yaml conf/test_times_2021.yaml:
    # that file is the full 256-timestamp test set, which would force a spatial panel for
    # every timestep. To highlight specific events, add e.g. --event 2021-07-24 (typhoons).

    echo "🚀 ===== ${region} / ${split} ====="
    apptainer exec --bind /home/atuin:/home/atuin "$CONTAINER" \
        python3 -u -m analysis.eval_pipeline.run compare $args \
            --outdir "$od" \
        || echo "⚠️  ${region}/${split} FAILED (continuing)"
}

echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"
# Controllable via env (sbatch --export=ALL,SPLITS=test,REGIONS=taiwan).
SPLITS="${SPLITS:-test year}"
REGIONS="${REGIONS:-taiwan europa}"
echo "SPLITS=[$SPLITS]  REGIONS=[$REGIONS]"
for split in $SPLITS; do
    for region in $REGIONS; do
        run_compare "$region" "$split"
    done
done
echo "✓ ALL DONE — $(date)"
