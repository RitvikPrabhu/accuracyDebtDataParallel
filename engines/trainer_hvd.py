import torch
from torch.utils.data import DataLoader, DistributedSampler
import horovod.torch as hvd

from data.loader import get_datasets
from engines.metrics import get_metrics


def train(model, optimizer, criterion, args):
    """Horovod training with manual allgather + torchmetrics on rank 0.

    For diagnostics (acc / macro F1 / ECE / confusion matrix), we:
      - compute logits/probs on each rank
      - hvd.allgather() probs and targets to all ranks
      - update torchmetrics only on rank 0 using gathered tensors
    """
    hvd_rank = hvd.rank()
    hvd_size = hvd.size()
    local_rank = hvd.local_rank()

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(local_rank)
    model.to(device)

    optimizer = hvd.DistributedOptimizer(
        optimizer,
        named_parameters=model.named_parameters(),
        compression=hvd.Compression.none,
    )
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    train_dataset, val_dataset = get_datasets()
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=hvd_size, rank=hvd_rank, shuffle=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    num_classes = 10

    if hvd_rank == 0:
        train_metrics = get_metrics(
            device="cpu", num_classes=num_classes, include_confmat=False
        )
        val_metrics = get_metrics(
            device="cpu", num_classes=num_classes, include_confmat=True
        )
    else:
        train_metrics = None
        val_metrics = None

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        model.train()
        if hvd_rank == 0:
            train_metrics.reset()

        # ----- training loop -----
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            probs = torch.softmax(logits, dim=-1).detach()
            gathered_probs = hvd.allgather(probs)
            gathered_targets = hvd.allgather(targets)

            if hvd_rank == 0:
                train_metrics.update(gathered_probs.cpu(), gathered_targets.cpu())

        if hvd_rank == 0:
            train_results = train_metrics.compute()
            train_acc = train_results["acc"].item()
            train_f1 = train_results["f1"].item()
            train_ece = train_results["ece"].item()
        else:
            train_acc = train_f1 = train_ece = None

        # ----- validation loop -----
        model.eval()
        if hvd_rank == 0:
            val_metrics.reset()

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                logits = model(inputs)
                probs = torch.softmax(logits, dim=-1)

                gathered_probs = hvd.allgather(probs)
                gathered_targets = hvd.allgather(targets)

                if hvd_rank == 0:
                    val_metrics.update(
                        gathered_probs.cpu(),
                        gathered_targets.cpu(),
                    )

        if hvd_rank == 0:
            val_results = val_metrics.compute()
            val_acc = val_results["acc"].item()
            val_f1 = val_results["f1"].item()
            val_ece = val_results["ece"].item()
            confmat = val_results["confmat"]

            print(
                f"[HVD] Epoch {epoch+1}/{args.epochs} "
                f"Train: acc={train_acc:.4f}, f1={train_f1:.4f}, ece={train_ece:.4f}  "
                f"Val: acc={val_acc:.4f}, f1={val_f1:.4f}, ece={val_ece:.4f}  "
                f"(size={hvd_size})"
            )
            # Confusion matrix for HVD is available as `confmat`.
