#!/bin/bash
#SBATCH -J quantom-hvd-multinode_4G_4G
#SBATCH -A HPCBIGDATA
#SBATCH -p a30_normal_q
#SBATCH -N 2
#SBATCH --ntasks-per-node=4    
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=2     
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

srun --mpi=pmix_v3 python -m quantom_ips.drivers.hvd_training_workflow \
    "environment.parser.path=[${DATA_PATH}]"
