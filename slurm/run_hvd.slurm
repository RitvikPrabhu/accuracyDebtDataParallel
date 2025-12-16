#!/bin/bash
#SBATCH -J hvd-1node-1GPUperNode-W1
#SBATCH -A HPCBIGDATA
#SBATCH -p a100_normal_q
#SBATCH -N 1
#SBATCH --ntasks-per-node=1    
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16     
#SBATCH -t 00:30:00
#SBATCH --output=outputs/slurm-%x-%j.out

set -euo pipefail

module load Miniconda3
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
source activate /home/ritvikp/.conda/envs/quantom/
cd ..
module load CUDA
module load OpenMPI

horovodrun -np $SLURM_GPUS_ON_NODE -H localhost:$SLURM_GPUS_ON_NODE \
  python main.py \
    --backend hvd \
    --model convex_logreg \
    --batch-size 32 \
    --lr 0.01 \
    --epochs 50
