#!/usr/bin/env bash
set -euo pipefail

# Single-process / single-GPU or CPU run
CFG=${1:-configs/experiments/r50_cifar10_single.yaml}

python runner.py -c "${CFG}"
