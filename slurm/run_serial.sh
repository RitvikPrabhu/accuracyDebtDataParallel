#!/bin/bash
#SBATCH -J debug-serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=outputs/serial-%j.out

set -euo pipefail

python main.py \
  --backend serial \
  --model convex_logreg \
  --batch-size 32 \
  --epochs 10
