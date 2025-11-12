#!/bin/bash
echo "--- ⬆️  Synchronisiere mit Cluster ... ---"

# --- Dynamische Pfade ---
# 1. Finde das Verzeichnis, in dem DIESES Skript liegt
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# 2. Der Projekt-Root ist eine Ebene HÖHER
LOKALER_PFAD=$( dirname "$SCRIPT_DIR" )

# 3. Die Ignore-Datei liegt NEBEN diesem Skript
EXCLUDE_FILE="$SCRIPT_DIR/.rsync-ignore"
# -------------------------

# --- Deine Cluster-Variablen ---
CLUSTER_USER="s373395"
CLUSTER_ADRESSE="julia2.hpc.uni-wuerzburg.de"
CLUSTER_PFAD="~/bjerknes"
# -------------------------

# Prüfen, ob die Ignore-Datei existiert
if [ ! -f "$EXCLUDE_FILE" ]; then
    echo "--- ⚠️ FEHLER: .rsync-ignore Datei nicht gefunden in $SCRIPT_DIR ---"
    exit 1
fi

rsync -avz --delete \
    --exclude-from "$EXCLUDE_FILE" \
    "$LOKALER_PFAD/" \
    "$CLUSTER_USER@$CLUSTER_ADRESSE:$CLUSTER_PFAD"

echo "--- ✅ Sicherer Upload fertig. ---"