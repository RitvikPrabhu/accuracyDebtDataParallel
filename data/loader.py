import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_datasets(data_root: str = None):
    """Return (train_dataset, val_dataset) for CIFAR-10.

    If DATA_ROOT is set in the environment, use that. Otherwise default to ./data.

    The first call will download CIFAR-10 if it is not already present.
    """
    if data_root is None:
        data_root = os.environ.get("DATA_ROOT", "./data")

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    cifar_dir = os.path.join(data_root, "cifar-10-batches-py")
    download_flag = not os.path.exists(cifar_dir)

    train_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=download_flag,
        transform=transform_train,
    )
    val_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=download_flag,
        transform=transform_test,
    )

    return train_dataset, val_dataset


def make_serial_dataloaders(batch_size: int, num_workers: int = 4):
    train_dataset, val_dataset = get_datasets()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
