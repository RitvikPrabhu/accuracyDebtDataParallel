# Ablation Runner

A config-driven experiment runner with pluggable datasets, models, backends, and distributed strategies (Torch DDP, Horovod, TF distribute stub). Includes CSV/TensorBoard/W&B loggers.

## Run (single-GPU / CPU)
```bash
python runner.py -c configs/experiments/r50_cifar10_ddp.yaml
```

## Run (multi-GPU, Torch DDP)
```bash
GPUS=4 bash scripts/launch.sh configs/experiments/r50_cifar10_ddp.yaml
```

## Switch strategy (Horovod)
```bash
python runner.py -c configs/experiments/r50_cifar10_horovod.yaml
```

## Toggle loggers via YAML
```yaml
logging:
  csv: {enabled: true, filename: metrics.csv}
  tensorboard: {enabled: true, log_dir: logs/myexp}
```
Open TensorBoard:
```bash
tensorboard --logdir logs/myexp
```
