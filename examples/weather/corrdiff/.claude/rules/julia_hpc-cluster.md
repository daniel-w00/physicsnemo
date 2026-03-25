# HPC Cluster Guidelines - Julia2

## Overview

**Cluster**: Julia2 HPC cluster at University of Würzburg
**Login**: `ssh <jmu_account>@julia2.hpc.uni-wuerzburg.de`
**Documentation**: See `docs/jullia2-docs/` for full details

The HPC cluster is used for:
- Model training (GPU-intensive)
- Large-scale data preprocessing
- Batch experiments with multiple hyperparameters

**Do NOT** use the cluster for:
- Small scripts or analysis (use local WSL instead)
- Running IDEs (VSCode, PyCharm, Cursor) on login nodes
- File inspection or data exploration
- Quick tests (run locally first, then use test partition)

## Cluster Resources

### Storage
- **Home**: `/home/<user>` - 150GB quota
  - Use for: code, environments, scripts
  - Total: 137TB on NVMEs (shared across all users)
- **Data**: `/data/<primary_group>/<user>` - no quota (currently)
  - Use for: datasets, checkpoints, large files
  - Total: 1.8PB on HDDs (shared across all users)

### Check Disk Usage
```bash
# Quick quota check
/usr/local/bin/getquota.sh

# Manual check (home directory)
getfattr -n ceph.dir.rbytes $HOME 2> /dev/null | grep ceph.dir.rbytes | grep -oP '(?<=\").*(?=\")' | numfmt --to=iec-i --suffix=B
```

### Login Node
- **CPUs**: 16 cores (users limited to 8 cores)
- **Memory**: 64GB total (users limited to 16GB)
- **Purpose**: Setup environment, transfer data, compile code, submit jobs
- **NOT for**: Running jobs directly, testing code, running IDEs

### Project Directories
- **Code**: `$HOME/alpha-earth-oro` (git repo)
- **Data**: `$HOME/data/` (large datasets)
- **Container**: `$HOME/apptainer/alpha-oro2.sif`

## SLURM Partitions

Use `sinfo` to view available partitions and their status.

| Partition | GPUs | Max CPUs/Threads | Time Limit | Use Case |
|-----------|------|------------------|------------|----------|
| **test** | 2 GPUs | 64 cores | 1 hour | Testing jobs before production |
| **standard** | 1-8 GPUs | 16 threads per GPU | 1 day | GPU training/inference (default for GPU work) |
| **small_cpu** | None | max 64 threads | 2 days | CPU-only jobs (default partition) |
| **large_cpu** | None | max 128 threads | 2 days | Large CPU-only jobs |
| **h100** | 1-8 H100s | 16 threads per GPU | 1 day | High-performance H100 GPUs (special use) |

### Default Settings
- **Default partition**: `small_cpu` (if not specified)
- **Default memory**: 2GB per thread (change with `--mem=XG`)
- **GPU partition**: Use `standard` for most GPU work

### GPU Selection

**Standard GPUs** (L40, L40s):
```bash
# Generic GPU request (1-8 GPUs)
#SBATCH --gres=gpu:1
#SBATCH -p standard

# Specific GPU model
#SBATCH --gres=gpu:L40:1      # 1-3 L40 per node
#SBATCH --gres=gpu:L40s:1     # 1-8 L40s per node
```

**H100 GPUs** (special use):
```bash
#SBATCH --gres=gpu:H100:1
#SBATCH -p h100
```

## SLURM Job Scripts

### Location
All cluster job scripts should be in `jobs/` directory.

### Template Structure
```bash
#!/bin/bash
#SBATCH -J JobName
#SBATCH -c 8                    # CPUs (16 threads per GPU for GPU jobs)
#SBATCH --mem=32G               # Memory (2G per thread is default)
#SBATCH --gres=gpu:1            # GPU count (1-8)
#SBATCH -p standard             # Partition (test, standard, h100, small_cpu, large_cpu)
#SBATCH --time=12:00:00         # Time limit (max: 1 day for GPU, 2 days for CPU)
#SBATCH --mail-type=ALL         # Email notifications
#SBATCH --mail-user=<your-email>
#SBATCH --output=output_jobname-%j.out

# Configuration
PROJECT_ROOT="$HOME/alpha-earth-oro"
CONTAINER_IMG="$HOME/apptainer/alpha-oro2.sif"
SCRIPT_PATH="$PROJECT_ROOT/path/to/script.py"
DATA_FILE="$HOME/data/path/to/data.nc"

# Load secrets (WANDB_API_KEY, etc.)
source ~/.job_secrets

# Environment setup
export WANDB_MODE=online

# Enable GPU support in container
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}

echo "Starting job on $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Data: $DATA_FILE"

# Execute with Apptainer
cd "$PROJECT_ROOT"

srun apptainer exec --nv \
    --bind "$PROJECT_ROOT":"$PROJECT_ROOT" \
    --bind "$HOME/data":"$HOME/data" \
    "$CONTAINER_IMG" \
    python3 -u "$SCRIPT_PATH" \
    --arg1 value1 \
    --arg2 value2

echo "Job completed."
```

### Current Training Job (train.slurm)
```bash
#!/bin/bash
#SBATCH -J TrainAlpha
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -p test                 # NOTE: Currently using test (1 hour limit)
#SBATCH --time=01:00:00         # Change partition to 'standard' for longer runs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daniel.wilhelm@stud-mail.uni-wuerzburg.de
#SBATCH --output=output_train-%j.out
```

**Important**: Current `jobs/train.slurm` uses `test` partition (1 hour limit). For real training:
- Change `-p test` to `-p standard`
- Increase `--time` to up to `1-00:00:00` (1 day)

## Submitting Jobs

### Submit a Job
```bash
sbatch jobs/train.slurm
# Output: Submitted batch job 204690
```

Output file will be created as `slurm-<JOBID>.out` in `$HOME` or `jobs/output_train-<JOBID>.out` if specified.

### Check Job Status
```bash
# Your jobs
squeue -u $USER

# All jobs on partition
squeue -p test

# Detailed job info
scontrol show job <JOBID>

# View partition status
sinfo
```

### Cancel a Job
```bash
scancel <JOBID>

# Cancel all your jobs
scancel -u $USER
```

### Monitor Output
```bash
# Watch output in real-time
tail -f jobs/output_train-<JOBID>.out

# Check completed job output
cat jobs/output_train-<JOBID>.out
```

### Interactive Sessions

**For testing and debugging**:
```bash
# Request interactive session
srun -c 2 -n 1 --mem=8G --gres=gpu:1 --tmp=5G -p test --pty bash

# With specific GPU
srun -c 8 --mem=16G --gres=gpu:L40:1 -p standard --time=02:00:00 --pty bash
```

**Important**:
- Test partition has 1 hour limit for interactive sessions
- Don't use interactive sessions for production (login node can reboot)
- Always test on test partition before submitting to standard

### Attach to Running Job for Debugging

**Prerequisites**: Setup internal SSH keys first (see Environment Setup section)

```bash
# Attach to running job
srun --pty --overlap --jobid <JOBID> /bin/bash
```

**Warning**: Do NOT submit jobs from within an interactive session - submit from login node instead.

## Data Management on Cluster

### Directory Structure
```
$HOME/
├── alpha-earth-oro/        # Project code (git repo)
├── data/                   # Data files
│   ├── hrrr_mini/
│   ├── alpha_earth/
│   └── processed/
├── apptainer/              # Container images
│   └── alpha-oro2.sif
└── .job_secrets            # API keys (WANDB_API_KEY, etc.)
```

### Best Practices
- **Code**: Keep code in git repo (`alpha-earth-oro/`)
- **Data**: Store large datasets in `$HOME/data/` (separate from code)
- **Checkpoints**: Save to `$HOME/data/checkpoints/` or project checkpoints/
- **Logs**: Job outputs go to `jobs/output_*.out`
- **Bind mounts**: Always bind both project and data directories in Apptainer

## Container Usage (Apptainer/Singularity)

Julia2 supports Apptainer (successor to Singularity) for containerized workflows.

### Auto-Mounted Directories
`/home` and `/data` are **automatically mounted** inside containers - no need to bind them explicitly!

### Running Commands in Container

**Interactive shell**:
```bash
# Shell with GPU support
apptainer shell --nv $HOME/apptainer/alpha-oro2.sif

# Binds are automatic for /home and /data
# Additional binds only needed for other directories
```

**Execute single command**:
```bash
# Run command in container
apptainer exec --nv $HOME/apptainer/alpha-oro2.sif python3 --version

# Run with explicit binds (usually not needed)
apptainer exec --nv \
    --bind "$HOME/alpha-earth-oro":"$HOME/alpha-earth-oro" \
    "$HOME/apptainer/alpha-oro2.sif" \
    python3 -u train.py
```

**In SLURM job** (recommended):
```bash
# Enable GPU visibility in container
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}

# Run with srun
srun apptainer exec --nv \
    --bind "$HOME/alpha-earth-oro":"$HOME/alpha-earth-oro" \
    --bind "$HOME/data":"$HOME/data" \
    "$HOME/apptainer/alpha-oro2.sif" \
    python3 -u train.py --args
```

### Important Flags
- `--nv`: Enable NVIDIA GPU support (required for GPU jobs)
- `--bind src:dest`: Mount host directory into container (optional - /home and /data auto-mounted)
- `-u` flag in `python3 -u`: Unbuffered output (see logs in real-time)

### Temporary Build Space and Cache

Apptainer uses custom locations for temporary files and cache on Julia2:

- **APPTAINER_TMPDIR**: `/data/<your_group>/<your_user>/.tmp/`
- **APPTAINER_CACHEDIR**: `/data/<your_group>/<your_user>/.cache/`

These directories may need to be created manually:
```bash
test -d $APPTAINER_TMPDIR || mkdir -p $APPTAINER_TMPDIR
test -d $APPTAINER_CACHEDIR || mkdir -p $APPTAINER_CACHEDIR
```

### Clean Up Cache
Run periodically to free disk space:
```bash
apptainer cache clean
```

## Wandb Setup

### Authentication
Store your Wandb API key in `~/.job_secrets`:

```bash
# ~/.job_secrets
export WANDB_API_KEY="your-api-key-here"
```

Source it in job scripts:
```bash
source ~/.job_secrets
export WANDB_MODE=online
```

### Offline Mode
If cluster has no internet during training:
```bash
export WANDB_MODE=offline
```

Then sync later:
```bash
wandb sync wandb/run-<timestamp>
```



### Python Virtual Environment Example

```bash
# On login node
mkdir project
cd project
python3 -m venv .
source bin/activate
pip3 install torch pytorch-lightning wandb xarray netcdf4 rasterio
```



## Running Multiple Experiments

### Array Jobs
Run multiple configurations in parallel:
```bash
#SBATCH --array=0-3  # Run 4 jobs (indices 0,1,2,3)

# Use SLURM_ARRAY_TASK_ID to vary parameters
MODELS=("linear" "simple" "current" "bigger")
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

srun apptainer exec --nv $CONTAINER python3 train.py --model $MODEL
```



### Attach to Running Job
**Prerequisites**: Internal SSH keys must be set up

```bash
srun --pty --overlap --jobid <JOBID> /bin/bash
```

### Common Issues


**Out of memory**
- Reduce batch size in training script
- Request more memory: `--mem=64G`
- Check actual usage: `sacct -j <JOBID> --format=JobID,MaxRSS,MaxVMSize`

**GPU not found**
- Verify `--gres=gpu:1` in SBATCH directives
- Check `--nv` flag in apptainer exec command
- Ensure `export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}` is set
- Test inside container: `nvidia-smi`

**Import errors**
- Verify container has required packages
- Rebuild container with missing dependencies
- Test: `apptainer exec <container> python3 -c "import package"`

**Submitting jobs from interactive session fails**
- Error: `srun: error: CPU binding outside of job step allocation`
- Solution: Submit jobs from login node, NOT from within interactive session

**Building container takes forever**
- This is normal - `mksquashfs` is slow
- Can take minutes to hours depending on image size
- Be patient or use existing containers when possible

**Disk quota exceeded**
- Check usage: `/usr/local/bin/getquota.sh`
- Clean apptainer cache: `apptainer cache clean`
- Move large datasets to `/data/<group>/<user>` instead of `$HOME`

## Best Practices Summary

1. **Always test locally in WSL first** before cluster submission
2. **Use test partition** for initial cluster testing (1 hour limit)
3. **Don't run IDEs** (VSCode, PyCharm, Cursor) on login nodes
4. **Store code in /home**, **data in /data** - respect quotas
5. **Clean cache periodically**: `apptainer cache clean`
6. **Monitor disk usage**: `/usr/local/bin/getquota.sh`
7. **Use wandb logging** to track experiments
8. **For production training**: Use `standard` partition with 1 day time limit
9. **Check job status**: `squeue -u $USER` before submitting many jobs
10. **Test with small data/few epochs** before full training runs

## Quick Reference


# Interactive session
srun -c 4 --mem=16G --gres=gpu:1 -p test --time=01:00:00 --pty bash

# Check disk usage
/usr/local/bin/getquota.sh

# Clean cache
apptainer cache clean
```
