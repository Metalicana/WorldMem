#!/bin/bash
#SBATCH --job-name=worldmem_mem
#SBATCH --output=logs/worldmem_mem_%j.out
#SBATCH --error=logs/worldmem_mem_%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

module load cuda
module load ffmpeg
module load anaconda
conda activate worldmem

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
mkdir -p "$HF_HOME"

WORLDMEM_ROOT="${WORLDMEM_ROOT:-$HOME/WorldMem}"
export WORLDMEM_ROOT
export WORLDMEM_STORAGE_ROOT="${WORLDMEM_STORAGE_ROOT:-$HOME/worldmem_results}"
export WORLDMEM_DATA_DIR="${WORLDMEM_DATA_DIR:-$WORLDMEM_ROOT/data/minecraft}"

mkdir -p "$WORLDMEM_ROOT/logs" "$WORLDMEM_STORAGE_ROOT"
cd "$WORLDMEM_ROOT"

bash scripts/run_worldmem_memory_policy_smoke.sh
