#!/bin/bash
#SBATCH -J taiwan-corrdiff
#SBATCH -c 16
#SBATCH -p standard
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.wilhelm@stud-mail.uni-wuerzburg.de
#SBATCH --output=output/job-%j.out
#SBATCH --tmp=1500G

source ~/.job_secrets

export WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
export APPTAINERENV_WANDB_API_KEY=${WANDB_API_KEY}
echo "CUDA Devices: ${CUDA_VISIBLE_DEVICES}"



# 1. Pfade
# Die ZIP-Datei, die du gefunden hast!
ARCHIVE_PATH="/data/42-julia-hpc-rz-lsx/sih25nq/downscaling/CorrDiff/cwa_dataset.zip"

# Das Haupt-Scratch-Verzeichnis auf der Node
LOCAL_DIR="$TMPDIR/corrdiff_${SLURM_JOB_ID}"
mkdir -p $LOCAL_DIR

echo "Lokales Verzeichnis: $LOCAL_DIR"


# 3. Daten kopieren (mit sauberem 60-Sekunden-Monitoring)
echo "Kopiere) auf lokale SSD..."
echo "Starte Kopiervorgang (cp) im Hintergrund."

# Starte das 'dumme' Kopieren im Hintergrund (&)
cp $ARCHIVE_PATH $LOCAL_DIR/cwa_dataset.zip &
CP_PID=$! # Speichere die Prozess-ID (PID) des Kopiervorgangs

echo "Kopiervorgang läuft (PID: $CP_PID). Überwache Fortschritt alle 60 Sekunden:"

# Schleife, die läuft, solange der cp-Prozess (mit der PID) existiert
# 'kill -0 $PID' prüft, ob der Prozesch läuft, ohne ihn zu beenden
while kill -0 $CP_PID 2>/dev/null; do
    # Gib die aktuelle Dateigröße der Zieldatei aus
    du -h $LOCAL_DIR/cwa_dataset.zip
    sleep 60
done

# 'wait' stellt sicher, dass das Skript erst weitergeht,
# wenn der cp-Prozess (CP_PID) 100% fertig ist.
wait $CP_PID

echo "Archiv kopiert."


# 4. Daten auf SSD entpacken
echo "Entpacke Archiv lokal auf SSD..."
unzip $LOCAL_DIR/cwa_dataset.zip -d $LOCAL_DIR
echo "Entpacken abgeschlossen."

# Optional: Lösche das Archiv, um SSD-Platz zu sparen
rm $LOCAL_DIR/cwa_dataset.zip

# --- ENDE: Lokale Daten einrichten ---

# Training starten
echo "Starte Training im Apptainer-Container..."


BIND_PATH="$LOCAL_DIR/cwa_dataset"


echo "Binde $BIND_PATH an /data im Container."

srun apptainer run --nv \
    --bind $BIND_PATH:/data \
    --bind ~/data/checkpoints/taiwan:/checkpoints/ \
    ~/apptainer/new_corrdiff.sif \
    python -u ~/bjerknes/train.py --config-name $1

echo "srun-Befehl beendet."

# Das 'trap' räumt jetzt $LOCAL_DIR auf (inkl. dem entpackten Ordner)