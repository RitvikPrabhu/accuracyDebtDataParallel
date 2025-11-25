import argparse, yaml
from core.config import load_config
from core.registry import DATASETS, MODELS, STRATEGIES, BACKENDS
from core.utils import set_seed
from trainloops.trainer import Trainer
from trainloops.hooks import CheckpointHook, CSVLogger, TensorBoardLogger, WandbLogger
from pathlib import Path
import torch


def resolve_group(path_or_group: str) -> dict:
    if path_or_group.endswith('.yaml'):
        with open(path_or_group, 'r') as f: 
            return yaml.safe_load(f)
    full = f"configs/{path_or_group}"
    with open(full, 'r') as f: 
        return yaml.safe_load(f)

def build_loggers(logging_cfg: dict, output_dir: str):
    hooks = []
    if logging_cfg.get("csv", {}).get("enabled", True):
        hooks.append(CSVLogger(outdir=output_dir, filename=logging_cfg["csv"].get("filename", "metrics.csv")))
    if logging_cfg.get("tensorboard", {}).get("enabled", False):
        hooks.append(TensorBoardLogger(outdir=logging_cfg["tensorboard"].get("log_dir", output_dir)))
    if logging_cfg.get("wandb", {}).get("enabled", False):
        wb = logging_cfg["wandb"]
        hooks.append(WandbLogger(project=wb.get("project", "ablation-runner"), name=wb.get("run_name"), config=wb.get("config")))
    return hooks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True, help='experiment yaml')
    args = ap.parse_args()
    cfg = load_config(args.config)

    if "app_cfg" in cfg:
        from applications.quantom_ips_adapter import QuantomIPSAdapter
        app_conf = cfg["app_cfg"]
        outdir = cfg.get("output_dir", f"logs/{cfg.get('experiment_name','external_app')}")
        Path(outdir).mkdir(parents=True, exist_ok=True)
        target = app_conf.get("target", "quantom_ips")
        if target != "quantom_ips":
            raise SystemExit(f"Unknown app adapter: {target}")
        code = QuantomIPSAdapter().run(app_conf, outdir)
        raise SystemExit(code)


    dataset_cfg = resolve_group(cfg['dataset_cfg'])
    model_cfg = resolve_group(cfg['model_cfg'])
    strategy_cfg = resolve_group(cfg['strategy_cfg'])
    backend_cfg = resolve_group(cfg['backend_cfg'])

    set_seed(cfg.get('seed', 1337))

    dataset = DATASETS.get(dataset_cfg['target'])(**{k:v for k,v in dataset_cfg.items() if k != 'target'})
    train_loader, num_classes = dataset.make(
        split=dataset_cfg.get('split','train'),
        batch_size=cfg['train']['batch_size'],
        num_workers=dataset_cfg.get('num_workers', 4)
    )

    model = MODELS.get(model_cfg['target'])(**{k:v for k,v in model_cfg.items() if k != 'target'}).build(num_classes)

    backend = BACKENDS.get(backend_cfg['target'])(**{k:v for k,v in backend_cfg.items() if k != 'target'})
    strategy = STRATEGIES.get(strategy_cfg['target'])(**{k:v for k,v in strategy_cfg.items() if k != 'target'})

    optim = torch.optim.SGD(model.parameters(), lr=cfg['train']['lr'], momentum=cfg['train']['momentum'], weight_decay=cfg['train']['weight_decay'])

    log_hooks = build_loggers(cfg.get('logging', {}), cfg['output_dir'])
    ckpt_hook = CheckpointHook(cfg['output_dir'], every_n_epochs=cfg.get('ckpt_interval',1))
    hooks = [*log_hooks, ckpt_hook]

    trainer = Trainer(strategy=strategy, backend=backend, hooks=hooks)
    trainer.fit(model, optim, train_loader, num_epochs=cfg['num_epochs'], log_interval=cfg.get('log_interval',50))

if __name__ == '__main__':
    main()
