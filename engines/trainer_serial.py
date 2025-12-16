import torch
from torch.utils.data import DataLoader
from data.loader import make_serial_dataloaders
from engines.metrics import get_metrics


def train(model, optimizer, criterion, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader, val_loader = make_serial_dataloaders(args.batch_size)

    num_classes = 10  # adjust if needed
    train_metrics = get_metrics(device, num_classes=num_classes, include_confmat=False)
    val_metrics = get_metrics(device, num_classes=num_classes, include_confmat=True)

    for epoch in range(args.epochs):
        model.train()
        train_metrics.reset()

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(inputs)
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
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                logits = model(inputs)
                probs = torch.softmax(logits, dim=-1)
                val_metrics.update(probs, targets)

        val_results = val_metrics.compute()
        val_acc = val_results["acc"].item()
        val_f1 = val_results["f1"].item()
        val_ece = val_results["ece"].item()
        confmat = val_results["confmat"]

        print(
            f"[Serial] Epoch {epoch+1}/{args.epochs} "
            f"Train: acc={train_acc:.4f}, f1={train_f1:.4f}, ece={train_ece:.4f}  "
            f"Val: acc={val_acc:.4f}, f1={val_f1:.4f}, ece={val_ece:.4f}"
        )

