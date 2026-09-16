#!/bin/bash
#SBATCH --job-name=eos48ue_binding_lipophilicity
#SBATCH --partition=sbnb_gpu_h200
#SBATCH --nodelist=irbgcn10
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/aloy/home/acomajuncosa/Ersilia/eos48ue/logs/binding_lipophilicity_%j.out
#SBATCH --error=/aloy/home/acomajuncosa/Ersilia/eos48ue/logs/binding_lipophilicity_%j.err

# 0-billing node only (sbnb_*) - never spot_* or irb_gpu_*, those cost the lab money.

set -euo pipefail
REPO_DIR=/aloy/home/acomajuncosa/Ersilia/eos48ue
cd "$REPO_DIR"

source ~/miniforge3/bin/activate eos48ue-train
python model/framework/fit/src/01_fit.py --cluster binding_lipophilicity
