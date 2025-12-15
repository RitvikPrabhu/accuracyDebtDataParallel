import os
import argparse
import torch

from models import (
    list_models,
    get_model_spec,
    build_model_from_name,
    build_loss_from_name,
    build_optimizer_from_name,
)
from engines import trainer_serial, trainer_ddp, trainer_hvd


def set_seeds(seed: int):
    """Set all relevant seeds for deterministic-ish runs."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        type=str,
        default="serial",
        choices=["serial", "ddp", "hvd"],
        help="Training backend: serial, ddp, or hvd.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="convex_logreg",
        choices=list_models(),
        help="Model name registered in models.registry.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Per-GPU batch size."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate (otherwise use model default).",
    )
    parser.add_argument("--local_rank", type=int, default=0, help="For DDP only.")
    args = parser.parse_args()

    # Backend setup & seeds
    if args.backend == "hvd":
        import horovod.torch as hvd

        hvd.init()
        torch.cuda.set_device(hvd.local_rank())
        set_seeds(42)
    elif args.backend == "ddp":
        import torch.distributed as dist

        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        args.local_rank = local_rank
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        set_seeds(42)
    else:
        set_seeds(42)

    # Build model / loss / optimizer from registry
    spec = get_model_spec(args.model)
    print(f"[{args.backend.upper()}] Using model '{spec.name}': {spec.description}")

    model = build_model_from_name(args.model)
    criterion = build_loss_from_name(args.model)
    optimizer = build_optimizer_from_name(args.model, model.parameters(), lr=args.lr)

    # Dispatch to engine
    if args.backend == "serial":
        trainer_serial.train(model, optimizer, criterion, args)
    elif args.backend == "ddp":
        trainer_ddp.train(model, optimizer, criterion, args)
    elif args.backend == "hvd":
        trainer_hvd.train(model, optimizer, criterion, args)


if __name__ == "__main__":
    main()
