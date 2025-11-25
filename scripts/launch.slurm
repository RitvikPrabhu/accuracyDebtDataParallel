#!/bin/bash
#SBATCH -J r50-c10
#SBATCH -A youracct
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -t 02:00:00

module load cuda/12.1
export OMP_NUM_THREADS=8
srun --mpi=pmix_v3 torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE runner.py -c configs/experiments/r50_cifar10_ddp.yaml
