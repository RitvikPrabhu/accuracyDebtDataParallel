#!/bin/bash
#SBATCH -J quantom-arar
#SBATCH -A HPCBIGDATA
#SBATCH -p a100_normal_q
#SBATCH -N 2
#SBATCH --ntasks-per-node=4        # 1 task per GPU
#SBATCH --gpus-per-node=4
#SBATCH -t 02:00:00

set -euo pipefail

module load cuda/12.1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

cd /home/ritvikp/git_repos/accuracyDebt_data_parallel_DL

DATA_PATH=${DATA_PATH:-applications/quantom-ips/data/data.npy}

srun --mpi=pmix_v3 --gpus-per-task=1 python -m quantom_ips.drivers.distributed_training_workflow \
  "environment.parser.path=[${DATA_PATH}]"
