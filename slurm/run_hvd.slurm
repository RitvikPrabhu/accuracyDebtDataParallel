#!/bin/bash
#SBATCH -J debug-hvd
#SBATCH --nodes=2                # X nodes
#SBATCH --ntasks-per-node=4      # Y GPUs per node
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=outputs/hvd-%j.out

set -euo pipefail

srun --mpi=pmi2 \
  python main.py \
    --backend hvd \
    --model bn_cnn_debug \
    --batch-size 32 \
    --lr 0.08 \
    --epochs 10
