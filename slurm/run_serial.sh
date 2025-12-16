#!/bin/bash
#SBATCH -J serial-1node-convexlogreg-B8
#SBATCH -A HPCBIGDATA
#SBATCH -p a100_normal_q
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -t 00:30:00
#SBATCH --output=outputs/slurm-%x-%j.out


set -euo pipefail

python main.py \
  --backend serial \
  --model convex_logreg \
  --batch-size 256 \
  --lr 0.01 \
  --epochs 50
