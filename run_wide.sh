#!/bin/bash
#SBATCH --job-name=p_wide
#SBATCH --account=def-schmidtm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24000M
#SBATCH --time=04:30:00
#SBATCH --array=0-170%16
#SBATCH --output=/home/gbaveja/scratch/plast_wide/logs/%x_%A_%a.out

set -uo pipefail
module load python/3.12
source ~/venv/continual/bin/activate
export WANDB_MODE=disabled

ROOT=$HOME/scratch/plast_wide
P=3
CHUNK=3
total=$(wc -l < "$ROOT/commands.txt")
start=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
running=0
for (( k=0; k<CHUNK; k++ )); do
  idx=$(( start + k )); [ "$idx" -ge "$total" ] && break
  line=$(( idx + 1 ))
  csv=$(sed -n "${line}p" "$ROOT/csvs.txt")
  cmd=$(sed -n "${line}p" "$ROOT/commands.txt")
  [ -s "$csv" ] && { echo "skip done $idx"; continue; }
  echo "launch $idx"; bash -c "$cmd" &
  running=$(( running + 1 ))
  if [ "$running" -ge "$P" ]; then wait -n; running=$(( running - 1 )); fi
done
wait
echo "chunk $SLURM_ARRAY_TASK_ID done"
