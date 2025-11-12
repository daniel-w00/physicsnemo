#!/bin/bash
echo "--- 🚀 Starte Hard-coded Job auf Cluster... ---"

# --- Deine Cluster-Variablen ---
CLUSTER_USER="s373395"
CLUSTER_ADRESSE="julia2.hpc.uni-wuerzburg.de"
CLUSTER_PFAD="~/bjerknes"
SLURM_SCRIPT_PATH="jobs/julia-d.slurm"

# --- HIER IST DEIN HARD-CODED CONFIG-NAME (KORRIGIERT: OHNE .yaml) ---
CONFIG_NAME="first_config_training_hrrr_mini_diffusion"
# ---------------------------------------------------------------------

echo "--- Starte Slurm-Job mit Config: $CONFIG_NAME ---"

# --- SCHRITT 1: Job starten und Job-ID einfangen ---

SBATCH_OUTPUT=$(ssh "$CLUSTER_USER@$CLUSTER_ADRESSE" "cd $CLUSTER_PFAD && sbatch $SLURM_SCRIPT_PATH $CONFIG_NAME")

if [[ $SBATCH_OUTPUT != "Submitted batch job "* ]]; then
    echo "--- ❌ FEHLER: Job-Einreichung fehlgeschlagen! ---"
    echo "Cluster-Antwort: $SBATCH_OUTPUT"
    exit 1
fi

JOB_ID=$(echo $SBATCH_OUTPUT | awk '{print $NF}')
echo "--- ✅ Job erfolgreich eingereicht. JOB ID: $JOB_ID ---"


# --- SCHRITT 2: Job-Status überwachen ---

echo ""
echo "--- ⏱️  Überwache Job $JOB_ID ... ---"
echo "--- 💡 HINWEIS: Drücke 'Strg+C' (Ctrl+C), um diese Überwachung abzubrechen. ---"
echo "--- (Der Job auf dem Cluster läuft trotzdem weiter!) ---"
echo ""

# Setze den Timer auf 0, JETZT, wo die Überwachung beginnt
SECONDS=0

# Prüfe den Status ein erstes Mal
JOB_STATUS=$(ssh "$CLUSTER_USER@$CLUSTER_ADRESSE" "squeue -j $JOB_ID -h")

# Falls der Job schon fertig ist
if [ -z "$JOB_STATUS" ]; then
    echo "--- 🎉 Job $JOB_ID ist bereits beendet. ---"
    exit 0
fi

# --- Phase 1: Schnelle Überwachung (solange $SECONDS < 50) ---
echo "--- Starte schnelle Überwachung (alle 5 Sekunden)... ---"

# Wir nutzen 'while' anstatt 'for', um die Zeit flexibel zu prüfen
while [ $SECONDS -lt 50 ]; do
    # Prüfen, ob der Job überhaupt noch läuft
    if [ -z "$JOB_STATUS" ]; then
        break # Job ist fertig, springe aus der 5s-Schleife raus
    fi

    JOB_STATE=$(echo $JOB_STATUS | awk '{print $5}')
    # Hier verwenden wir $SECONDS für die Zeitangabe
    echo "   ... Status: $JOB_STATE (Zeit: ~${SECONDS}s)"

    sleep 5

    JOB_STATUS=$(ssh "$CLUSTER_USER@$CLUSTER_ADRESSE" "squeue -j $JOB_ID -h")
done


# --- Phase 2: Normale Überwachung (alle 30s) ---

# Prüfen, ob der Job nach Phase 1 noch läuft
if [ -z "$JOB_STATUS" ]; then
    echo "--- 🎉 Job $JOB_ID ist beendet (während der schnellen Phase). ---"
    exit 0
fi

echo "--- Wechsle zu normaler Überwachung (alle 30 Sekunden)... ---"
while [ -n "$JOB_STATUS" ]; do
    JOB_STATE=$(echo $JOB_STATUS | awk '{print $5}')
    # Hier verwenden wir $SECONDS für die Zeitangabe
    echo "   ... Status: $JOB_STATE (Zeit: ~${SECONDS}s)"

    sleep 30

    JOB_STATUS=$(ssh "$CLUSTER_USER@$CLUSTER_ADRESSE" "squeue -j $JOB_ID -h")
done

echo "--- 🎉 Job $JOB_ID ist beendet (Zeit: ~${SECONDS}s). ---"