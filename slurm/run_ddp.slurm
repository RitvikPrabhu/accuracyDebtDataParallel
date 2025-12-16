#!/bin/bash
#SBATCH -J ddp-1node-1GPUperNode-convexlogreg-S4
#SBATCH -A HPCBIGDATA
#SBATCH -p a100_normal_q
#SBATCH -N 1
#SBATCH --ntasks-per-node=1        
#SBATCH --gpus-per-node=8
#SBATCH -t 00:30:00
#SBATCH --output=outputs/slurm-%x-%j.out

set -euo pipefail

module load Miniconda3
source activate /home/ritvikp/.conda/envs/quantom/
module load CUDA

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
DEFAULT_PORT=$((29500 + ${SLURM_JOBID:-0} % 1000))
export MASTER_PORT=${MASTER_PORT:-$DEFAULT_PORT}
export OMP_NUM_THREADS=8  
echo "Num GPUs per node: $SLURM_GPUS_PER_NODE"
cd ..

# export NUM_GPUS_PER_NODE=$SLURM_GPUS_PER_NODE

srun torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node="$SLURM_GPUS_PER_NODE" \
  --rdzv_id="$SLURM_JOB_ID" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  main.py \
    --backend ddp \
    --model convex_logreg \
    --batch-size 32 \
    --lr 0.01 \
    --epochs 50
