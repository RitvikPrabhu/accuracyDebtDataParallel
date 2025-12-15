import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from data.loader import get_datasets
from engines.metrics import get_metrics


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def train(model, optimizer, criterion, args):
    """DDP training with SyncBatchNorm and torchmetrics diagnostics."""
    local_rank = args.local_rank
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(local_rank)

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    train_dataset, val_dataset = get_datasets()
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    num_classes = 10
    train_metrics = get_metrics(device, num_classes=num_classes, include_confmat=False)
    val_metrics = get_metrics(device, num_classes=num_classes, include_confmat=True)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        ddp_model.train()
        train_metrics.reset()

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = ddp_model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            probs = torch.softmax(logits, dim=-1)
            train_metrics.update(probs, targets)

        train_results = train_metrics.compute()
        train_acc = train_results["acc"].item()
        train_f1 = train_results["f1"].item()
        train_ece = train_results["ece"].item()

        # ----- validation -----
        val_metrics.reset()
        ddp_model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                logits = ddp_model(inputs)
                probs = torch.softmax(logits, dim=-1)
                val_metrics.update(probs, targets)

        val_results = val_metrics.compute()
        val_acc = val_results["acc"].item()
        val_f1 = val_results["f1"].item()
        val_ece = val_results["ece"].item()
        confmat = val_results["confmat"]

        if is_main_process():
            print(
                f"[DDP] Epoch {epoch+1}/{args.epochs} "
                f"Train: acc={train_acc:.4f}, f1={train_f1:.4f}, ece={train_ece:.4f}  "
                f"Val: acc={val_acc:.4f}, f1={val_f1:.4f}, ece={val_ece:.4f}  "
                f"(world_size={world_size})"
            )
            # Confusion matrix for DDP is available as `confmat`.
