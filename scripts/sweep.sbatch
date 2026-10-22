#!/bin/bash
# Hyperparameter sweep on a SLURM GPU cluster (e.g. UGA GACRC Sapelo2).
# One array task = one grid combination from configs/sweep.yaml (72 total).
#
#   sbatch scripts/sweep.sbatch                 # all 72 runs, 8 at a time
#   sbatch --array=0-3 scripts/sweep.sbatch     # just the first 4
#   uv run python src/sweep.py --collect        # afterwards: ranked reports/sweep_results.csv
#
# Check your cluster's partition and GPU names (`sinfo`) and edit the two lines below.
#SBATCH --job-name=cg-sweep
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-71%8
#SBATCH --output=logs/sweep-%A_%a.out

set -euo pipefail
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

# uv installs into the project's .venv; HF downloads go to scratch, not home quota.
export HF_HOME="${SCRATCH:-$HOME}/hf_cache"
export MLFLOW_DISABLE_AGENT_HINT=1
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv sync --extra ml
uv run python src/sweep.py --index "$SLURM_ARRAY_TASK_ID"
