#!/bin/bash
#SBATCH -J quantom-ddp-multinode_3G_3G
#SBATCH -N 2
#SBATCH -A HPCBIGDATA
#SBATCH -p a30_normal_q
#SBATCH --ntasks-per-node=4   
#SBATCH --gpus-per-node=4      
#SBATCH --cpus-per-task=4
#SBATCH -t 10:00:00
#SBATCH --output=outputs/slurm-%x-%j.out

set -euo pipefail

module load Miniconda3
source activate /home/ritvikp/.conda/envs/quantom/
module load CUDA/12.6.0

cd ..

DATA_PATH=${DATA_PATH:-applications/quantom-ips/data/data.npy}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

srun bash -c '
 export RANK=$SLURM_PROCID
 export LOCAL_RANK=$SLURM_LOCALID
 export WORLD_SIZE=$SLURM_NTASKS
 
 echo "Rank $RANK (Local $LOCAL_RANK) on $(hostname)"

 python -u -m quantom_ips.drivers.ddp_training_workflow \
    "environment.parser.path=['"${DATA_PATH}"']"
'