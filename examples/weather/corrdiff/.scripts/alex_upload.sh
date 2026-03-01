#!/bin/bash
echo "--- Synchronisiere mit Cluster Alex ... ---"

# --- Dynamische Pfade ---
# 1. Finde das Verzeichnis, in dem DIESES Skript liegt
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# 2. Der Projekt-Root ist eine Ebene HÖHER
LOKALER_PFAD=$( dirname "$SCRIPT_DIR" )

# 3. Die Ignore-Datei liegt NEBEN diesem Skript
EXCLUDE_FILE="$SCRIPT_DIR/.rsync-ignore"
# -------------------------

# --- Cluster-Variablen (nutzt SSH-Alias "alex") ---
CLUSTER_HOST="alex"
CLUSTER_PFAD="~/corrdiff"
# -------------------------

# Prüfen, ob die Ignore-Datei existiert
if [ ! -f "$EXCLUDE_FILE" ]; then
    echo "--- FEHLER: .rsync-ignore Datei nicht gefunden in $SCRIPT_DIR ---"
    exit 1
fi

rsync -avz --delete \
    --exclude-from "$EXCLUDE_FILE" \
    "$LOKALER_PFAD/" \
    "$CLUSTER_HOST:$CLUSTER_PFAD"

echo "--- Upload zu Alex fertig. ---"
