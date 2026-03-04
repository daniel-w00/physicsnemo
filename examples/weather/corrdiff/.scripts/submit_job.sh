#!/bin/bash
echo "--- 🚀 Starte Job auf Cluster... ---"

# --- SCHRITT 1: CLUSTER-AUSWAHL ---
CLUSTER_CHOICE="$1"

if [ -z "$CLUSTER_CHOICE" ]; then
    echo "--- EINGABE ERFORDERLICH ---"
    echo "Verfügbare Cluster: julia, alex"
    read -p "Welcher Cluster soll genutzt werden?: " CLUSTER_CHOICE
fi

# Cluster-spezifische Konfigurationen
if [ "$CLUSTER_CHOICE" == "alex" ]; then
    SSH_TARGET="alex"
    CLUSTER_PFAD="~/corrdiff"
elif [ "$CLUSTER_CHOICE" == "julia" ] || [ "$CLUSTER_CHOICE" == "julia2" ]; then
    SSH_TARGET="s373395@julia2.hpc.uni-wuerzburg.de"
    CLUSTER_PFAD="~/bjerknes"
else
    echo "--- ❌ FEHLER: Unbekannter Cluster '$CLUSTER_CHOICE'. Erlaubt sind 'julia' oder 'alex'. ---"
    exit 1
fi

SLURM_SCRIPT_DIR="jobs"
echo "--- 🎯 Ziel: $CLUSTER_CHOICE ($SSH_TARGET) im Pfad $CLUSTER_PFAD ---"

# --- SCHRITT 2: SLURM-SKRIPT (Pflicht) ---
SLURM_SCRIPT_NAME="$2"

if [ -z "$SLURM_SCRIPT_NAME" ]; then
    read -p "Name des Slurm-Skripts (z.B. job.slurm): " SLURM_SCRIPT_NAME
fi

if [ -z "$SLURM_SCRIPT_NAME" ]; then
    echo "--- ❌ FEHLER: Slurm-Skript darf nicht leer sein! ---"
    exit 1
fi

# --- SCHRITT 3: CONFIG-NAME (Optional) ---
CONFIG_NAME="$3"

# Nur interaktiv fragen, wenn das Skript komplett ohne Parameter (oder nur mit Cluster) aufgerufen wurde
if [ -z "$3" ] && [ -z "$2" ]; then
    read -p "Name der Config (Optional, Enter zum Überspringen): " CONFIG_NAME
fi
# ---------------------------------------------------------------------

echo ""
if [ -z "$CONFIG_NAME" ]; then
    echo "--- Starte Slurm-Job: $SLURM_SCRIPT_NAME (OHNE Config) ---"
    SBATCH_CMD="cd $CLUSTER_PFAD && sbatch $SLURM_SCRIPT_DIR/$SLURM_SCRIPT_NAME"
else
    echo "--- Starte Slurm-Job: $SLURM_SCRIPT_NAME mit Config: $CONFIG_NAME ---"
    SBATCH_CMD="cd $CLUSTER_PFAD && sbatch $SLURM_SCRIPT_DIR/$SLURM_SCRIPT_NAME $CONFIG_NAME"
fi

# --- SCHRITT 4: Job starten und Job-ID einfangen ---
SBATCH_OUTPUT=$(ssh "$SSH_TARGET" "$SBATCH_CMD")

if [[ $SBATCH_OUTPUT != "Submitted batch job "* ]]; then
    echo "--- ❌ FEHLER: Job-Einreichung fehlgeschlagen! ---"
    echo "Cluster-Antwort: $SBATCH_OUTPUT"
    exit 1
fi

JOB_ID=$(echo "$SBATCH_OUTPUT" | awk '{print $NF}')
echo "--- ✅ Job erfolgreich eingereicht. JOB ID: $JOB_ID ---"

# --- SCHRITT 5: Status und Log-Pfad EINMALIG abfragen ---
echo "--- 🔍 Hole Job-Details... ---"
# Wir holen Queue-Infos und den Log-Pfad in einem einzigen SSH-Aufruf
JOB_INFO=$(ssh "$SSH_TARGET" "
    squeue -j $JOB_ID -h -o '%T|%r|%S'
    scontrol show job $JOB_ID | grep 'StdOut=' | cut -d'=' -f2
")

# Ausgabe parsen
JOB_STATE=$(echo "$JOB_INFO" | sed -n '1p' | cut -d'|' -f1)
JOB_REASON=$(echo "$JOB_INFO" | sed -n '1p' | cut -d'|' -f2)
JOB_START=$(echo "$JOB_INFO" | sed -n '1p' | cut -d'|' -f3)
LOG_PATH=$(echo "$JOB_INFO" | sed -n '2p')

echo "   ... Status: $JOB_STATE"
if [ "$JOB_STATE" == "PENDING" ]; then
    echo "   ... Grund für Warten: $JOB_REASON"
    echo "   ... Geplanter Start:  $JOB_START"
fi
echo "   ... Erwarteter Log-Pfad: $LOG_PATH"

# --- SCHRITT 6: Verbinden und warten ---
echo ""
echo "--- 📺 Verbinde zum Live-Log... ---"
echo "--- 💡 HINWEIS: Drücke 'Strg+C', um die Ansicht zu beenden. Der Job läuft weiter! ---"
echo "---------------------------------------------------"

# Phase 1: Warte bis RUNNING; Phase 2: Warte auf Log-Datei; Phase 3: Stream bis Job fertig
ssh -o ServerAliveInterval=60 -t "$SSH_TARGET" "

    # --- Phase 1: Warte bis Job RUNNING ist ---
    echo '⏳ Warte auf Job-Start...'
    CHECKS=0
    while true; do
        STATE=\$(squeue -j $JOB_ID -h -o '%T|%r|%S' 2>/dev/null)
        if [ -z \"\$STATE\" ]; then
            echo '⚠️  Job nicht mehr in Queue (möglicherweise sofort beendet).'
            break
        fi
        JOB_STATUS=\$(echo \"\$STATE\" | cut -d'|' -f1)
        JOB_REASON=\$(echo \"\$STATE\" | cut -d'|' -f2)
        JOB_START=\$(echo \"\$STATE\" | cut -d'|' -f3)
        if [ \"\$JOB_STATUS\" == 'RUNNING' ]; then
            echo '✅ Job läuft!'
            break
        fi
        echo \"   ... \$JOB_STATUS | Grund: \$JOB_REASON | Geplanter Start: \$JOB_START\"
        CHECKS=\$((CHECKS + 1))
        if [ \$CHECKS -lt 10 ]; then
            sleep 3   # erste ~30s: alle 3s prüfen
        else
            sleep 40  # danach: alle 40s prüfen
        fi
    done

    # --- Phase 2: Warte auf Log-Datei ---
    echo '⏳ Warte auf Log-Datei...'
    while [ ! -f \"$LOG_PATH\" ]; do
        sleep 5
        if ! squeue -j $JOB_ID -h 2>/dev/null | grep -q .; then
            [ -f \"$LOG_PATH\" ] && break
            echo '⚠️  Job nicht mehr in Queue und Log-Datei fehlt. Abbruch.'
            exit 1
        fi
    done

    echo '✅ Log-Datei gefunden! Starte Stream...'
    echo '---------------------------------------------------'

    # --- Phase 3: Stream bis Job fertig ---
    tail -n 50 -f \"$LOG_PATH\" &
    TAIL_PID=\$!
    trap 'kill \$TAIL_PID 2>/dev/null' EXIT

    while squeue -j $JOB_ID -h 2>/dev/null | grep -q .; do
        sleep 100
    done

    sleep 3  # kurz warten damit letzte Log-Zeilen noch ankommen
    kill \$TAIL_PID 2>/dev/null
    wait \$TAIL_PID 2>/dev/null
"

echo "---------------------------------------------------"
echo "--- 👋 Stream beendet. ---"