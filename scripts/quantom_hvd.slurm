#!/bin/bash
#SBATCH -J quantom-hvd-1node-2GPUperNode
#SBATCH -A HPCBIGDATA
#SBATCH -p a100_normal_q
#SBATCH -N 1
#SBATCH --ntasks-per-node=2    
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16     
#SBATCH -t 10:00:00
#SBATCH --output=outputs/slurm-%x-%j.out

set -euo pipefail


module load Miniconda3
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
source activate /home/ritvikp/.conda/envs/quantom/
cd ..
module load CUDA
module load OpenMPI

DATA_PATH=${DATA_PATH:-applications/quantom-ips/data/data.npy}

# srun --mpi=pmix_v3 --gpus-per-task=1 python -m quantom_ips.drivers.hvd_training_workflow \
#   "environment.parser.path=[${DATA_PATH}]"

horovodrun -np $SLURM_GPUS_ON_NODE -H localhost:$SLURM_GPUS_ON_NODE \
    python -m quantom_ips.drivers.hvd_training_workflow \
    "environment.parser.path=[${DATA_PATH}]"
