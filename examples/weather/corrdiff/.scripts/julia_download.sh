#!/bin/bash
echo "--- ⬇️  Lade Änderungen vom Cluster ... ---"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
LOKALER_PFAD=$( dirname "$SCRIPT_DIR" )

CLUSTER_USER="s373395"
CLUSTER_ADRESSE="julia2.hpc.uni-wuerzburg.de"
CLUSTER_PFAD="~/bjerknes"

rsync -avz \
    --exclude=".git/" \
    --exclude="__pycache__/" \
    --exclude="*.pyc" \
    --exclude="*.egg-info/" \
    --exclude=".venv/" \
    --exclude="wandb/" \
    --exclude="tensorboard/" \
    --exclude=".claude/" \
    --exclude="checkpoints/" \
    "$CLUSTER_USER@$CLUSTER_ADRESSE:$CLUSTER_PFAD/" \
    "$LOKALER_PFAD/"

echo "--- ✅ Download fertig. ---"
