#!/usr/bin/env bash
set -euo pipefail

# Horovod multi-process launch
CFG=${1:-configs/experiments/r50_cifar10_horovod.yaml}
NP=${NP:-4}

horovodrun -np "${NP}" python runner.py -c "${CFG}"
