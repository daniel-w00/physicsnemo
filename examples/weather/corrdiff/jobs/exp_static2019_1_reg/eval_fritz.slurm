#!/bin/bash -l
#SBATCH -J exp_static2019_1_reg_eval
#SBATCH -A b214cb
#SBATCH --ntasks=1
#SBATCH -c 72
#SBATCH --time=24:00:00
#SBATCH --partition=singlenode
#SBATCH --output=output/fritz/jobout/eval-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user "daniel.wilhelm@stud-mail.uni-wuerzburg.de"

# Fritz (NHR@FAU CPU cluster) copy of eval.slurm — no GPU.
# The eval_pipeline is pure CPU work (torch CPU backend), so Fritz's CPU-only
# singlenode partition fits it directly; the Alex version only requested a GPU
# because Alex has no CPU partition.
#
# Fritz gives a WHOLE exclusive node (72 cores / 250 GB) per job. To use it
# efficiently we run all (region × split) compares CONCURRENTLY on the one node,
# splitting the cores between them, then wait — instead of idling half the node
# running them one at a time.
#
# Scope: BOTH regions, both splits (europa/taiwan × year/test). Each compare
# overlays the 6 trained runs PLUS the pure-regression baseline = 7 models;
# the baseline is vault/generated/europa_baseline/reg_eu_pure_v1 for europa and
# $GEN/taiwan_baseline for taiwan. Truth is overlaid automatically by the
# pipeline (first model is the truth/grid reference).
#   4 compares in parallel, ~18 threads each.
# Override with e.g.  sbatch --export=ALL,SPLITS=year,REGIONS=europa ...
# A compare is SKIPPED if it already has new-pipeline output (compare_qq.png);
# old-pipeline results (metrics_comparison.csv only) are recomputed in place —
# same numbers, plus the new plots (delta heatmap, error maps, cycles, Q-Q,
# RAPSD ratio). FINDINGS.md files in the result dirs are left untouched.
#
# NOTE: io.py swaps the first stored channel by region
# (maximum_radar_reflectivity ↔ precipitation_amount_1hr via _resolve_channels),
# so europa compares run fine — the old "europa fails" note no longer applies.
#
# AD-HOC MODE: set MODELS to run ONE explicit compare of arbitrary files instead
# of the region×split sweep above (the whole node goes to that single compare).
# MODELS is a ';'-separated list of  name=path  pairs; the first is the reference
# (truth / grid / event-timestep source). OUTDIR overrides the result dir.
#   sbatch --export=ALL,\
#   MODELS="diff_eu_v1=$HPCVAULT/generated/diff_eu_v1-test2021.nc;static2019_alpha_concat_N1=$WORK/generated/exp_static2019_1_diff/europa_alpha_concat_N1/test2021.nc",\
#   OUTDIR=analysis/eval_pipeline/results/diff_eu_v1_vs_static2019_alpha_concat_N1 \
#       jobs/exp_static2019_1_reg/eval_fritz.slurm

set -u
cd "$HOME/corrdiff"

CONTAINER=$HOME/apptainer/corrdiff_zarr3.sif
GEN=$WORK/generated/exp_static2019_1_reg
BASELINE_DIR=$HPCVAULT/generated/europa_baseline/reg_eu_pure_v1
RUNS="alpha_concat_N1 alpha_branch_N1 olmo_concat_N1 olmo_branch_N1 alpha_branch_N8 olmo_branch_N8"

NCORES=${SLURM_CPUS_PER_TASK:-72}

run_compare () {  # $1=region  $2=split  $3=threads  $4=logfile
    local region=$1 split=$2 threads=$3 log=$4 args=""
    local od="analysis/eval_pipeline/results/${region}_${split}"

    if ls "$od"/*/plots/compare_qq.png >/dev/null 2>&1; then
        echo "⏭️  skip ${region}/${split} (new-pipeline results already exist in $od)"
        return 0
    fi

    for r in $RUNS; do
        args="$args --pred $GEN/${region}_${r}/v1/${split}2021.nc --name ${r}"
    done
    # Pure-regression baseline, appended as the last overlaid model. Must keep
    # the name "baseline": the delta heatmap / ΔRMSE maps reference it by name.
    if [ "$region" = "europa" ]; then
        args="$args --pred $BASELINE_DIR/reg_eu_pure_v1-step2000128-${split}2021.nc --name baseline"
    else
        args="$args --pred $GEN/taiwan_baseline/v1/${split}2021.nc --name baseline"
    fi
    # Default sample panels (n_samples=3). To highlight specific events add e.g.
    # --event 2021-07-24 (typhoons); do NOT pass the full 256-stamp test YAML.

    echo "🚀 ===== ${region} / ${split}  (${threads} threads) → $log ====="
    OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads \
    apptainer exec --bind /home/atuin:/home/atuin --bind /home/vault:/home/vault "$CONTAINER" \
        python3 -u -m analysis.eval_pipeline.run compare $args \
            --outdir "$od" \
        > "$log" 2>&1 \
        && echo "✅ ${region}/${split} done" \
        || echo "⚠️  ${region}/${split} FAILED (see $log)"
}

run_generic_compare () {  # uses $MODELS (name=path;...), the whole node, into $OUTDIR
    local args="" spec name path od
    local OLDIFS=$IFS IFS=';'
    for spec in $MODELS; do
        name=${spec%%=*}; path=${spec#*=}
        [ -n "$name" ] && [ -n "$path" ] && [ "$name" != "$path" ] \
            || { echo "❌ bad MODELS entry '$spec' (want name=path)" >&2; exit 1; }
        [ -f "$path" ] || { echo "❌ missing input: $path" >&2; exit 1; }
        args="$args --pred $path --name $name"
    done
    IFS=$OLDIFS
    od="${OUTDIR:-analysis/eval_pipeline/results/adhoc_compare_${SLURM_JOB_ID}}"
    echo "🚀 generic compare ($NCORES threads) → $od"
    OMP_NUM_THREADS=$NCORES MKL_NUM_THREADS=$NCORES \
    apptainer exec --bind /home/atuin:/home/atuin --bind /home/vault:/home/vault "$CONTAINER" \
        python3 -u -m analysis.eval_pipeline.run compare $args --outdir "$od" \
        && echo "✅ generic compare done → $od" \
        || { echo "⚠️  generic compare FAILED" >&2; exit 1; }
}

echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"

# Ad-hoc single compare of explicit files — bypasses the region×split sweep.
if [ -n "${MODELS:-}" ]; then
    run_generic_compare
    echo "✓ ALL DONE — $(date)"
    exit 0
fi

SPLITS="${SPLITS:-year test}"
REGIONS="${REGIONS:-europa taiwan}"

# Build the work list, then split cores evenly across the concurrent compares.
JOBS=()
for split in $SPLITS; do
    for region in $REGIONS; do
        JOBS+=("$region $split")
    done
done
NJOBS=${#JOBS[@]}
THREADS=$(( NCORES / NJOBS )); [ "$THREADS" -lt 1 ] && THREADS=1
echo "SPLITS=[$SPLITS]  REGIONS=[$REGIONS]  → $NJOBS concurrent compare(s), $THREADS threads each"

LOGDIR="output/fritz/jobout"
for spec in "${JOBS[@]}"; do
    set -- $spec
    run_compare "$1" "$2" "$THREADS" "$LOGDIR/eval-${SLURM_JOB_ID}-${1}_${2}.log" &
done
wait
echo "✓ ALL DONE — $(date)"
