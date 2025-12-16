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

module load Miniconda3
source activate /home/ritvikp/.conda/envs/quantom/
module load CUDA

echo "Num GPUs per node: $SLURM_GPUS_PER_NODE"
cd ..

python main.py \
  --backend serial \
  --model convex_logreg \
  --batch-size 256 \
  --lr 0.01 \
  --epochs 50
