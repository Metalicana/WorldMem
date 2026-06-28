#!/bin/bash
#SBATCH --job-name=worldmem_grid
#SBATCH --output=logs/worldmem_grid_%A_%a.out
#SBATCH --error=logs/worldmem_grid_%A_%a.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-27%4

set -euo pipefail

module load cuda
module load ffmpeg
module load anaconda
conda activate worldmem

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
mkdir -p "$HF_HOME"

WORLDMEM_REPO_ROOT="${WORLDMEM_REPO_ROOT:-$HOME/WorldMem}"
export WORLDMEM_REPO_ROOT
export WORLDMEM_STORAGE_ROOT="${WORLDMEM_STORAGE_ROOT:-$HOME/worldmem_results}"
export WORLDMEM_DATA_DIR="${WORLDMEM_DATA_DIR:-$WORLDMEM_REPO_ROOT/data/minecraft}"

NUM_VIDEOS="${NUM_VIDEOS:-30}"
CONTEXT_FRAMES="${CONTEXT_FRAMES:-600}"
FPS="${FPS:-10}"
SAMPLING_TIMESTEPS="${SAMPLING_TIMESTEPS:-20}"
export NUM_VIDEOS CONTEXT_FRAMES FPS SAMPLING_TIMESTEPS

mkdir -p "$WORLDMEM_REPO_ROOT/logs" "$WORLDMEM_STORAGE_ROOT"
cd "$WORLDMEM_REPO_ROOT"

declare -a COMBOS=()
for duration in 10 20 30 60; do
  COMBOS+=("unbounded none $duration")
  for policy in fifo rarity_irreplaceability slam_covisibility; do
    for budget in 32 64; do
      COMBOS+=("$policy $budget $duration")
    done
  done
done

combo="${COMBOS[$SLURM_ARRAY_TASK_ID]}"
read -r policy budget duration <<< "$combo"

export MEMORY_POLICY="$policy"
export FUTURE_SECONDS="$duration"
if [ "$budget" = "none" ]; then
  unset MEMORY_BUDGET
  export RUN_NAME="worldmem_${policy}_${duration}s_n${NUM_VIDEOS}"
else
  export MEMORY_BUDGET="$budget"
  export RUN_NAME="worldmem_${policy}_b${budget}_${duration}s_n${NUM_VIDEOS}"
fi

bash scripts/run_worldmem_memory_policy_smoke.sh
