#!/bin/bash
#SBATCH -J quantom-ddp
#SBATCH -A HPCBIGDATA
#SBATCH -p a100_normal_q
#SBATCH -N 2
#SBATCH --ntasks-per-node=4        
#SBATCH --gpus-per-node=4
#SBATCH -t 02:00:00

set -euo pipefail

module load cuda/12.1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

cd /home/ritvikp/git_repos/accuracyDebt_data_parallel_DL

DATA_PATH=${DATA_PATH:-applications/quantom-ips/data/data.npy}

# Map SLURM env to torch.distributed expectations.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29500}
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# One process per GPU across nodes.
srun --mpi=pmix_v3 --gpus-per-task=1 python -m quantom_ips.drivers.ddp_training_workflow \
  "environment.parser.path=[${DATA_PATH}]"
