#!/bin/bash
echo "--- ⬇️  Lade Änderungen von Alex ... ---"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
LOKALER_PFAD=$( dirname "$SCRIPT_DIR" )

CLUSTER_HOST="alex"
CLUSTER_PFAD="~/corrdiff"

rsync -avz \
    --exclude=".git/" \
    --exclude="__pycache__/" \
    --exclude="*.pyc" \
    --exclude="*.egg-info/" \
    --exclude=".venv/" \
    --exclude="wandb/" \
    --exclude="tensorboard/" \
    --exclude=".claude/settings.local.json" \
    --exclude="checkpoints/" \
    "$CLUSTER_HOST:$CLUSTER_PFAD/" \
    "$LOKALER_PFAD/"

echo "--- ✅ Download von Alex fertig. ---"
