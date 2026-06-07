#!/bin/bash
# Submit driver for exp_static2019_1_reg (12 regression runs, static-2019 embeddings).
#
# Per run it submits a 4-GPU training job and two dependent (afterok) 1-GPU
# generate jobs (test split + full-year set). The --constraint flag is added
# ONLY for the four N8 runs (a100_80); N1 runs take any A100 (40 or 80 GB).
#
# Usage:
#   ./submit_experiment.sh [VERSION] [run ...|all] [--gen-only]
#
#   VERSION     checkpoint/output version token (default: v1)
#   run ...     one or more run ids (default / "all" = every run)
#   --gen-only  submit just the two generate jobs (no train, no dependency) —
#               for (re)generating after training already finished.
#
# Examples:
#   ./submit_experiment.sh v1 taiwan_alpha_concat_N1
#   ./submit_experiment.sh v1 all
#   ./submit_experiment.sh v1 taiwan_olmo_branch_N8 --gen-only
set -euo pipefail

EXP=exp_static2019_1_reg
JOBDIR=$HOME/corrdiff/jobs/$EXP

ALL_RUNS=(
  taiwan_alpha_concat_N1 taiwan_alpha_branch_N1 taiwan_olmo_concat_N1 taiwan_olmo_branch_N1
  europa_alpha_concat_N1 europa_alpha_branch_N1 europa_olmo_concat_N1 europa_olmo_branch_N1
  taiwan_alpha_branch_N8 taiwan_olmo_branch_N8 europa_alpha_branch_N8 europa_olmo_branch_N8
)

# region per run
declare -A REGION
for r in "${ALL_RUNS[@]}"; do
  case "$r" in
    taiwan_*) REGION[$r]=cwb ;;
    europa_*) REGION[$r]=europa ;;
  esac
done

# TRAIN constraint per run: a100_80 for N8 AND for *_branch_N1.
#   - N8: ~6.6 GB resident field + branch -> needs 80 GB.
#   - branch_N1: the emb-branch (compile OFF per plan) peaks ~55 GB -> OOMs on a
#     40 GB card (observed 2026-06-06). concat_N1 is fine on 40 GB (compile ON,
#     no branch, ~24 GB), so it stays unconstrained.
# GENERATE constraint: a100_80 only for N8 (unchanged); single-GPU regression
# inference is light, so branch_N1 generate does NOT need the constraint.
declare -A TR_CONSTRAINT GEN_CONSTRAINT
for r in "${ALL_RUNS[@]}"; do
  if [[ "$r" == *_N8 || "$r" == *_branch_N1 ]]; then
    TR_CONSTRAINT[$r]=a100_80
  else
    TR_CONSTRAINT[$r]=""
  fi
  if [[ "$r" == *_N8 ]]; then
    GEN_CONSTRAINT[$r]=a100_80
  else
    GEN_CONSTRAINT[$r]=""
  fi
done

# --- parse args ---
VERSION=v1
GEN_ONLY=0
RUNS=()
for arg in "$@"; do
  case "$arg" in
    --gen-only) GEN_ONLY=1 ;;
    all)        RUNS=("${ALL_RUNS[@]}") ;;
    v[0-9]*)    VERSION="$arg" ;;
    *)          RUNS+=("$arg") ;;
  esac
done
if [ ${#RUNS[@]} -eq 0 ]; then
  RUNS=("${ALL_RUNS[@]}")
fi

echo "VERSION=$VERSION  gen_only=$GEN_ONLY  runs: ${RUNS[*]}"

for run in "${RUNS[@]}"; do
  if [ -z "${REGION[$run]:-}" ]; then
    echo "❌ unknown run: $run" >&2; exit 1
  fi
  reg=${REGION[$run]}
  CKPT_DIR=/checkpoints/$EXP/$run/$VERSION
  OUT_DIR=/generated/$EXP/$run/$VERSION
  TRC=${TR_CONSTRAINT[$run]:+--constraint=${TR_CONSTRAINT[$run]}}
  GENC=${GEN_CONSTRAINT[$run]:+--constraint=${GEN_CONSTRAINT[$run]}}

  if [ "$GEN_ONLY" -eq 1 ]; then
    echo "[$run] gen-only (test + year), constraint='${GEN_CONSTRAINT[$run]}'"
    sbatch $GENC -J ${run}_gtest \
      --export=ALL,REGION=$reg,GEN_CFG=gen_${run},CKPT_DIR=$CKPT_DIR,OUT_FILE=$OUT_DIR/test2021.nc,TIMES_MODE=test \
      $JOBDIR/generate.slurm
    sbatch $GENC -J ${run}_gyear \
      --export=ALL,REGION=$reg,GEN_CFG=gen_${run},CKPT_DIR=$CKPT_DIR,OUT_FILE=$OUT_DIR/year2021.nc,TIMES_MODE=year \
      $JOBDIR/generate.slurm
    continue
  fi

  echo "[$run] train + 2 generate (afterok), constraint='${TR_CONSTRAINT[$run]}'"
  tjid=$(sbatch --parsable -J ${run}_tr $TRC \
      --export=ALL,REGION=$reg,TRAIN_CFG=train_${run},CKPT_DIR=$CKPT_DIR,NUM_GPUS=4 \
      $JOBDIR/train.slurm)
  echo "   train job: $tjid"
  sbatch --dependency=afterok:$tjid $GENC -J ${run}_gtest \
      --export=ALL,REGION=$reg,GEN_CFG=gen_${run},CKPT_DIR=$CKPT_DIR,OUT_FILE=$OUT_DIR/test2021.nc,TIMES_MODE=test \
      $JOBDIR/generate.slurm
  sbatch --dependency=afterok:$tjid $GENC -J ${run}_gyear \
      --export=ALL,REGION=$reg,GEN_CFG=gen_${run},CKPT_DIR=$CKPT_DIR,OUT_FILE=$OUT_DIR/year2021.nc,TIMES_MODE=year \
      $JOBDIR/generate.slurm
done
