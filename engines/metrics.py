import torchmetrics


def get_metrics(device, num_classes: int = 10, include_confmat: bool = False):
    """Return a MetricCollection for acc / macro F1 / ECE (and optional confmat).

    We feed probabilities (softmax outputs) and integer labels.
    For DDP, torchmetrics will use torch.distributed to sync states on compute().
    For Horovod, we only run metrics on rank 0 with manually gathered data.
    """
    metric_dict = {
        "acc": torchmetrics.classification.MulticlassAccuracy(
            num_classes=num_classes
        ),
        "f1": torchmetrics.classification.MulticlassF1Score(
            num_classes=num_classes,
            average="macro",
        ),
        "ece": torchmetrics.classification.MulticlassCalibrationError(
            num_classes=num_classes,
            n_bins=15,
        ),
    }

    if include_confmat:
        metric_dict["confmat"] = (
            torchmetrics.classification.MulticlassConfusionMatrix(
                num_classes=num_classes
            )
        )

    metrics = torchmetrics.MetricCollection(metric_dict)
    return metrics.to(device)
