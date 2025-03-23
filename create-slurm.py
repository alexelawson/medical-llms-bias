#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import math
import os

def create_slurm_script(file_path, num_chunks=8):
    """Create a SLURM array job script."""
    
    # Load the dataset
    print(f"Loading dataset from {file_path}")
    df = pd.read_parquet(file_path)
    
    # Filter single choice questions if needed
    df = df[df["choice_type"] == "single"]
    total_rows = len(df)
    
    # Calculate chunk size
    chunk_size = math.ceil(total_rows / num_chunks)
    
    # Create the SLURM script
    slurm_script = """#!/bin/bash
#SBATCH --job-name=llm_eval
#SBATCH --output=logs/llm_eval_%A_%a.out
#SBATCH --error=logs/llm_eval_%A_%a.err
#SBATCH --array=0-{}
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# Create logs directory if it doesn't exist
mkdir -p logs

# Calculate start and end indices
chunk_size={}
total_rows={}
start_idx=$((SLURM_ARRAY_TASK_ID * chunk_size))
end_idx=$(((SLURM_ARRAY_TASK_ID + 1) * chunk_size))

# Ensure end_idx doesn't exceed total rows
if [ $end_idx -gt $total_rows ]; then
    end_idx=$total_rows
fi

# Activate your conda environment if needed
# conda activate your_env_name

# Run the script
python llm-prompt.py \\
    --start $start_idx \\
    --end $end_idx \\
    --data {} \\
    --output-dir ./results \\
    --cache-dir ./cache
""".format(
    num_chunks - 1,  # array range is 0 to num_chunks-1
    chunk_size,
    total_rows,
    file_path
)
    
    # Write the script to a file
    script_path = "run_llm_eval.sh"
    with open(script_path, "w") as f:
        f.write(slurm_script)
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    
    print(f"\nCreated SLURM array job script: {script_path}")
    print(f"Total rows: {total_rows}")
    print(f"Number of chunks: {num_chunks}")
    print(f"Approximate chunk size: {chunk_size}")
    print("\nTo submit the job, run:")
    print("sbatch run_llm_eval.sh")

if __name__ == "__main__":
    create_slurm_script("augmented_dataset.parquet")