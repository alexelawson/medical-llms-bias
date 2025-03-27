#!/bin/bash
#SBATCH --job-name=llm_eval
#SBATCH --output=logs/llm_eval_%A_%a.out
#SBATCH --error=logs/llm_eval_%A_%a.err
#SBATCH --array=0-7
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# Create logs directory if it doesn't exist
mkdir -p logs

# Calculate start and end indices
chunk_size=150957
total_rows=1207650
start_idx=$((SLURM_ARRAY_TASK_ID * chunk_size))
end_idx=$(((SLURM_ARRAY_TASK_ID + 1) * chunk_size))

# Ensure end_idx doesn't exceed total rows
if [ $end_idx -gt $total_rows ]; then
    end_idx=$total_rows
fi

# Activate your conda environment if needed
# conda activate your_env_name

# Run the script
python llm-prompt.py \
    --start $start_idx \
    --end $end_idx \
    --data augmented_dataset.parquet \
    --output-dir ./results \
    --cache-dir ./cache
