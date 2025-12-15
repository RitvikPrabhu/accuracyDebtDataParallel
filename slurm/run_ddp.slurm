#!/bin/bash
#SBATCH -J ddp-1node-1GPUperNode
#SBATCH -A HPCBIGDATA
#SBATCH -p a100_normal_q
#SBATCH -N 1
#SBATCH --ntasks-per-node=1        
#SBATCH --gpus-per-node=1
#SBATCH -t 10:00:00
#SBATCH --output=outputs/slurm-%x-%j.out

set -euo pipefail

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NNODES=${SLURM_NNODES}, NGPUS_PER_NODE=${SLURM_NTASKS_PER_NODE}"

srun torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node="$SLURM_NTASKS_PER_NODE" \
  --rdzv_id="$SLURM_JOB_ID" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  main.py \
    --backend ddp \
    --model bn_cnn_debug \
    --batch-size 32 \
    --lr 0.08 \
    --epochs 10
