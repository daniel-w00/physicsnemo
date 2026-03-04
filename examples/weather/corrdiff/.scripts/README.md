# submit_job.sh

Reicht einen SLURM-Job auf dem Cluster ein und streamt das Log live.

## Features

- Einreichen per SSH mit einem Befehl vom lokalen Rechner
- Zeigt Job-Status und geplanten Startzeit (PENDING-Phase)
- Wartet automatisch bis der Job startet, dann Live-Stream des Logs
- Bricht sauber ab wenn der Job vor dem Log-Start scheitert
- `Strg+C` beendet nur den Stream — der Job läuft weiter

## Verwendung

```bash
.scripts/submit_job.sh <cluster> <slurm-skript> [config]
```

| Argument | Pflicht | Beschreibung |
|----------|---------|--------------|
| `cluster` | ja | `alex` oder `julia` |
| `slurm-skript` | ja | Dateiname in `jobs/` (z.B. `train.slurm`) |
| `config` | nein | Config-Name, wird als Argument an `sbatch` weitergegeben |

Ohne Argumente aufgerufen fragt das Skript interaktiv nach.

## Beispiele

```bash
.scripts/submit_job.sh alex train.slurm
.scripts/submit_job.sh alex gen_alex-d.slurm config_generate_taiwan
```

## Anpassung für andere Nutzer

In `submit_job.sh` oben die Cluster-Konfiguration anpassen:

```bash
if [ "$CLUSTER_CHOICE" == "alex" ]; then
    SSH_TARGET="alex"                  # SSH-Alias oder user@host
    CLUSTER_PFAD="~/corrdiff"          # Projektpfad auf dem Cluster
elif [ "$CLUSTER_CHOICE" == "julia" ]; then
    SSH_TARGET="s373395@julia2..."     # Eigenen Account eintragen
    CLUSTER_PFAD="~/bjerknes"
fi
```

Voraussetzung: SSH-Key-basierter Zugang zum Cluster (kein Passwort-Prompt).
