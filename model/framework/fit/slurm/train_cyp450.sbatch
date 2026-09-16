#!/bin/bash
#SBATCH --job-name=eos48ue_cyp450
#SBATCH --partition=sbnb_gpu_3090
#SBATCH --nodelist=irbgcn07
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/aloy/home/acomajuncosa/Ersilia/eos48ue/logs/cyp450_%j.out
#SBATCH --error=/aloy/home/acomajuncosa/Ersilia/eos48ue/logs/cyp450_%j.err

# 0-billing node only (sbnb_*) - never spot_* or irb_gpu_*, those cost the lab money.

set -euo pipefail
REPO_DIR=/aloy/home/acomajuncosa/Ersilia/eos48ue
cd "$REPO_DIR"

source ~/miniforge3/bin/activate eos48ue-train
python model/framework/fit/src/01_fit.py --cluster cyp450
