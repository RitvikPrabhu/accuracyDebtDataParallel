#!/bin/bash
#SBATCH -J quantom-ddp-1node-4GPUperNode
#SBATCH -A HPCBIGDATA
#SBATCH -p a30_normal_q
#SBATCH -N 1
#SBATCH --ntasks-per-node=4    
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16     
#SBATCH -t 10:00:00
#SBATCH --output=outputs/slurm-%x-%j.out

set -euo pipefail

module load Miniconda3
source activate /home/ritvikp/.conda/envs/quantom/
module load CUDA/12.6.0

cd ..

DATA_PATH=${DATA_PATH:-applications/quantom-ips/data/data.npy}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
DEFAULT_PORT=$((29500 + ${SLURM_JOBID:-0} % 1000))
export MASTER_PORT=${MASTER_PORT:-$DEFAULT_PORT}
export OMP_NUM_THREADS=8  

srun torchrun \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
        --nnodes=1 \
            --rdzv_id=$SLURM_JOB_ID \
                --rdzv_backend=c10d \
                    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
                        -m quantom_ips.drivers.ddp_training_workflow \
                            "environment.parser.path=[${DATA_PATH}]"