#!/bin/bash
#SBATCH -J quantom-ddp-mpi-1node-1GPUperNode
#SBATCH -A HPCBIGDATA
#SBATCH -p a100_normal_q
#SBATCH -N 1
#SBATCH --ntasks-per-node=1        
#SBATCH --gpus-per-node=1
#SBATCH -t 10:00:00
#SBATCH --output=outputs/slurm-%x-%j.out

set -euo pipefail

module load Miniconda3
source activate /home/ritvikp/.conda/envs/quantom/

module load OpenMPI
module load CUDA/12.6.0

cd ..

DATA_PATH=${DATA_PATH:-applications/quantom-ips/data/data.npy}
export OMPI_MCA_pml=ob1 
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_opal_warn_on_missing_libcuda=0

srun --mpi=pmix_v3 --gpus-per-task=1 python -m quantom_ips.drivers.ddp_mpi_training_workflow \
  "environment.parser.path=[${DATA_PATH}]"
