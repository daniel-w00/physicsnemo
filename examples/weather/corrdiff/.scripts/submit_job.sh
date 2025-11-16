#!/bin/bash
echo "--- 🚀 Starte Job auf Cluster... ---"

# --- Deine Cluster-Variablen ---
CLUSTER_USER="s373395"
CLUSTER_ADRESSE="julia2.hpc.uni-wuerzburg.de"
CLUSTER_PFAD="~/bjerknes"
SLURM_SCRIPT_DIR="jobs" # Nur das Verzeichnis, der Dateiname kommt variabel

# --- ⚠️ HINWEIS ZU PYCHARM ---
# Wenn du das Skript in PyCharm OHNE Argumente startest,
# wird es bei 'read' hängenbleiben, da die Konsole
# dort oft keine interaktiven Eingaben erlaubt!
#
# EMPFEHLUNG: Trage die Argumente in PyCharm in der
# "Run Configuration" -> "Program arguments" ein.
# z.B.: julia-d.slurm first_config_training
# ---


# --- SCHRITT 0: SLURM-SKRIPT-NAME (Hybrid-Ansatz) ---
if [ -n "$1" ]; then
    # Fall 1: Argument 1 wurde übergeben
    SLURM_SCRIPT_NAME="$1"
    echo "--- Info: Verwende Slurm-Skript aus Argument 1: $SLURM_SCRIPT_NAME"
else
    # Fall 2: Argument 1 fehlt, frage interaktiv
    echo "--- EINGABE ERFORDERLICH ---"
    read -p "Name des Slurm-Skripts (z.B. julia-d.slurm): " SLURM_SCRIPT_NAME
fi


# --- SCHRITT 0: CONFIG-NAME (Hybrid-Ansatz) ---
if [ -n "$2" ]; then
    # Fall 1: Argument 2 wurde übergeben
    CONFIG_NAME="$2"
    echo "--- Info: Verwende Config-Name aus Argument 2: $CONFIG_NAME"
else
    # Fall 2: Argument 2 fehlt, frage interaktiv
    if [ -z "$1" ]; then
        # Nur anzeigen, wenn wir schon im interaktiven Modus sind
        echo "--- EINGABE ERFORDERLICH ---"
    fi
    read -p "Name der Config (z.B. first_config_...): " CONFIG_NAME
fi

# --- Gültigkeitsprüfung ---
if [ -z "$SLURM_SCRIPT_NAME" ] || [ -z "$CONFIG_NAME" ]; then
    echo "--- ❌ FEHLER: Slurm-Skript oder Config-Name sind leer! ---"
    exit 1
fi
# ---------------------------------------------------------------------

echo ""
echo "--- Starte Slurm-Job: $SLURM_SCRIPT_NAME mit Config: $CONFIG_NAME ---"

# --- SCHRITT 1: Job starten und Job-ID einfangen ---
# Die Variablen werden hier kombiniert:
SBATCH_OUTPUT=$(ssh "$CLUSTER_USER@$CLUSTER_ADRESSE" "cd $CLUSTER_PFAD && sbatch $SLURM_SCRIPT_DIR/$SLURM_SCRIPT_NAME $CONFIG_NAME")

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