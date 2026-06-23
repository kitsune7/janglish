#!/bin/bash
#SBATCH --partition=utucset-gpu-grn
#SBATCH --qos=utucset-gpu-grn
#SBATCH --account=larsenc
#SBATCH --job-name=janglish
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=janglish-%j.out
#SBATCH --error=janglish-%j.err

set -euo pipefail

# Slurm executes a spooled copy of this script, so BASH_SOURCE points into
# Slurm's spool directory. Use the directory where sbatch was invoked instead.
SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
cd "${SCRIPT_DIR}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo ""

# set up environment
module load uv
module load python/3.12.4

# verify gpu is available
nvidia-smi -L || true

uv run python3 cmd/train/train_loop.py
