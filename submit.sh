#!/bin/bash

conda activate /ubc/cs/research/kmyi/jeffyct/never-loses-forge/envs/medrag

export HF_HOME=./.cache/huggingface
export TORCH_HOME=./.cache/torch

python test.py
