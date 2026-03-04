# submit_job.sh

Reicht einen SLURM-Job auf dem Cluster ein und streamt das Log live.

## Verwendung

```bash
.scripts/submit_job.sh <cluster> <slurm-skript> [config]
```

| Argument | Pflicht | Beschreibung |
|----------|---------|--------------|
| `cluster` | ja | `alex` oder `julia` |
| `slurm-skript` | ja | Dateiname in `jobs/` (z.B. `train.slurm`) |
| `config` | nein | Config-Name, wird als Argument an `sbatch` weitergegeben |

## Beispiele

```bash
# Training ohne Config
.scripts/submit_job.sh alex train.slurm

# Generation mit Config
.scripts/submit_job.sh alex gen_alex-d.slurm config_generate_taiwan

# Interaktiv (fragt nach Eingaben)
.scripts/submit_job.sh
```

Nach der Einreichung wartet das Skript bis der Job startet, dann streamt es das Log live.
Mit `Strg+C` wird nur der Stream beendet — der Job läuft weiter.
