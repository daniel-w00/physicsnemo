#!/bin/bash
echo "--- ⬇️  Lade Logs UND neuesten Checkpoint... ---"

# --- Deine Cluster-Variablen ---
CLUSTER_USER="s373395"
CLUSTER_ADRESSE="julia2.hpc.uni-wuerzburg.de"
CLUSTER_PROJEKT_PFAD="~/bjerknes"

# --- Namen der Ergebnis-Ordner (basierend auf unseren Logs) ---
LOG_DIR_NAME="output"
CHECKPOINT_DIR_NAME="checkpoints_regression"
# -----------------------------------------------------------------

# --- Dynamische Pfade ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
LOKALER_PROJEKT_PFAD=$( dirname "$SCRIPT_DIR" )
# -------------------------

# --- ZIEL 1: Der 'output' Ordner (Logs) ---
# Das ist schnell, da Logs klein sind. Wir spiegeln sie komplett.
echo "--- [1/2] Synchronisiere '$LOG_DIR_NAME' Ordner (Logs)... ---"
LOKALES_ZIEL_OUTPUT="$LOKALER_PROJEKT_PFAD/$LOG_DIR_NAME/"
CLUSTER_QUELLE_OUTPUT="$CLUSTER_PROJEKT_PFAD/$LOG_DIR_NAME/"
mkdir -p "$LOKALES_ZIEL_OUTPUT"
rsync -avz --delete \
    "$CLUSTER_USER@$CLUSTER_ADRESSE:$CLUSTER_QUELLE_OUTPUT" \
    "$LOKALES_ZIEL_OUTPUT"


# --- ZIEL 2: Der "smarte" Checkpoint-Download ---
echo "--- [2/2] Suche NEUESTEN Checkpoint in '$CHECKPOINT_DIR_NAME'... ---"
CLUSTER_CHECKPOINT_PFAD="$CLUSTER_PROJEKT_PFAD/$CHECKPOINT_DIR_NAME"
LOKALES_ZIEL_CHECKPOINT_PFAD="$LOKALER_PROJEKT_PFAD/$CHECKPOINT_DIR_NAME/"
mkdir -p "$LOKALES_ZIEL_CHECKPOINT_PFAD"
echo "--- Suche in Cluster-Ordner: $CLUSTER_CHECKPOINT_PFAD ---"

# HIER IST DIE MAGIE:
# 1. 'ls -t ...' listet Dateien nach Zeit (neueste zuerst)
# 2. 'head -n 1' nimmt nur die erste Zeile
# 3. Wir suchen nur nach *.mdlus (das Modell), nicht *.pt (Optimizer)
LATEST_FILE=$(ssh "$CLUSTER_USER@$CLUSTER_ADRESSE" "ls -t $CLUSTER_CHECKPOINT_PFAD/*.mdlus | head -n 1")

# Prüfen, ob wir eine Datei gefunden haben
if [ -z "$LATEST_FILE" ]; then
    echo "--- ⚠️ HINWEIS: Konnte keine '.mdlus'-Dateien in $CLUSTER_CHECKPOINT_PFAD finden. Überspringe Checkpoint-Download. ---"
else
    echo "--- ✅ Neueste Datei gefunden: $LATEST_FILE ---"
    echo "--- Lade NUR diese eine Datei herunter... ---"

    # rsync --progress zeigt einen Ladebalken
    rsync -avz --progress \
        "$CLUSTER_USER@$CLUSTER_ADRESSE:$LATEST_FILE" \
        "$LOKALES_ZIEL_CHECKPOINT_PFAD"

    echo "--- Optional: Lösche alte lokale Checkpoints... ---"
    # Dieses 'find' löscht alle .mdlus-Dateien im lokalen Ordner,
    # die NICHT dem Namen der neuesten Datei entsprechen.
    find "$LOKALES_ZIEL_CHECKPOINT_PFAD" -type f -name "*.mdlus" ! -name "$(basename "$LATEST_FILE")" -exec rm {} +
    echo "--- Lokaler Ordner ist aufgeräumt. ---"
fi

echo "--- 🎉 Download fertig. ---"