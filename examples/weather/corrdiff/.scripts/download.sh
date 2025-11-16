#!/bin/bash
echo "--- ⬇️  Lade Logs UND neueste Checkpoints (rekursiv pro Basis-Ordner)... ---"

# --- Config: Deine Cluster-Variablen ---
CLUSTER_USER="s373395"
CLUSTER_ADRESSE="julia2.hpc.uni-wuerzburg.de"
CLUSTER_PROJEKT_PFAD="~/bjerknes" # (z.B. /home/s373395/bjerknes)
LOG_DIR_NAME="output"

# !!! WICHTIG: Alle Basis-Ordner, die Checkpoints enthalten könnten
# (Das Skript durchsucht jeden dieser Ordner und alle seine Unterordner)
CHECKPOINT_BASE_PATHS=(
    "~/data/checkpoints"
)
# -----------------------------------------------------------------

# --- Dynamische Pfade (Lokal auf deinem PC) ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
LOKALER_PROJEKT_PFAD=$( dirname "$SCRIPT_DIR" )
LOKALES_ZIEL_LOGS="$LOKALER_PROJEKT_PFAD/$LOG_DIR_NAME/"
# -------------------------

# --- ZIEL 1: Der 'output' Ordner (Logs) ---
# Das ist schnell, da Logs klein sind. Wir spiegeln sie komplett.
echo "--- [1/2] Synchronisiere Logs... ---"
CLUSTER_QUELLE_LOGS="$CLUSTER_PROJEKT_PFAD/$LOG_DIR_NAME/"

echo "    Cluster-Quelle: $CLUSTER_USER@$CLUSTER_ADRESSE:$CLUSTER_QUELLE_LOGS"
echo "    Lokales-Ziel:   $LOKALES_ZIEL_LOGS"

mkdir -p "$LOKALES_ZIEL_LOGS"
rsync -avz --delete \
    "$CLUSTER_USER@$CLUSTER_ADRESSE:$CLUSTER_QUELLE_LOGS" \
    "$LOKALES_ZIEL_LOGS"


# --- ZIEL 2: Lade neueste Checkpoints aus JEDEM Basis-Ordner ---
echo "--- [2/2] Suche neueste .mdlus und .pt in jedem Basis-Ordner... ---"

for CLUSTER_BASE_PATH in "${CHECKPOINT_BASE_PATHS[@]}"; do
    echo ""
    echo "--- Verarbeite Basis-Ordner: $CLUSTER_BASE_PATH ---"

    # Erstelle das lokale Zielverzeichnis (z.B. .../corrdiff/checkpoints_regression/)
    BASE_NAME=$(basename "$CLUSTER_BASE_PATH")
    LOKALES_ZIEL_SUBFOLDER="$LOKALER_PROJEKT_PFAD/$BASE_NAME/"
    mkdir -p "$LOKALES_ZIEL_SUBFOLDER"
    echo "    Lokales Ziel: $LOKALES_ZIEL_SUBFOLDER"

    # --- Finde und lade .mdlus ---
    # HIER IST DIE NEUE MAGIE:
    # 'find ... -type f -name "*.mdlus"' : Findet ALLE .mdlus Dateien rekursiv
    # '-exec ls -t {} +' : Übergibt alle Treffer an 'ls -t' (sortiert nach Zeit)
    # 'head -n 1' : Nimmt die oberste (neueste) Datei
    CMD_MDLUS="find $CLUSTER_BASE_PATH -type f -name \"*.mdlus\" -exec ls -t {} + 2>/dev/null | head -n 1"

    # Führe den Befehl auf dem Cluster aus
    LATEST_MDLUS=$(ssh "$CLUSTER_USER@$CLUSTER_ADRESSE" "$CMD_MDLUS")

    if [ -z "$LATEST_MDLUS" ]; then
        echo "    Keine .mdlus Datei in $CLUSTER_BASE_PATH gefunden (rekursiv). Überspringe."
    else
        echo "    Lade neueste .mdlus: $LATEST_MDLUS"
        rsync -avz --progress \
            "$CLUSTER_USER@$CLUSTER_ADRESSE:$LATEST_MDLUS" \
            "$LOKALES_ZIEL_SUBFOLDER"

        # Lokal aufräumen: Lösche alle .mdlus außer der, die wir gerade geholt haben
        find "$LOKALES_ZIEL_SUBFOLDER" -type f -name "*.mdlus" ! -name "$(basename "$LATEST_MDLUS")" -exec rm {} +
    fi

    # --- Finde und lade .pt (gleiche Logik) ---
    CMD_PT="find $CLUSTER_BASE_PATH -type f -name \"*.pt\" -exec ls -t {} + 2>/dev/null | head -n 1"
    LATEST_PT=$(ssh "$CLUSTER_USER@$CLUSTER_ADRESSE" "$CMD_PT")

    if [ -z "$LATEST_PT" ]; then
        echo "    Keine .pt Datei in $CLUSTER_BASE_PATH gefunden (rekursiv). Überspringe."
    else
        echo "    Lade neueste .pt: $LATEST_PT"
        rsync -avz --progress \
            "$CLUSTER_USER@$CLUSTER_ADRESSE:$LATEST_PT" \
            "$LOKALES_ZIEL_SUBFOLDER"

        # Lokal aufräumen: Lösche alle .pt außer der, die wir gerade geholt haben
        find "$LOKALES_ZIEL_SUBFOLDER" -type f -name "*.pt" ! -name "$(basename "$LATEST_PT")" -exec rm {} +
    fi
done

echo ""
echo "--- Lokale Checkpoint-Ordner sind aufgeräumt. ---"
echo "--- 🎉 Download fertig. ---"