#!/usr/bin/env bash
set -euo pipefail
CFG=${1:-configs/experiments/r50_cifar10_ddp.yaml}
GPUS=${GPUS:-4}

torchrun --nnodes=1 --nproc_per_node=${GPUS}   --master_port=${MASTER_PORT:-29500}   runner.py -c ${CFG}
