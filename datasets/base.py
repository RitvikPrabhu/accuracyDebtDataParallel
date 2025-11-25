from typing import Protocol

class DatasetFactory(Protocol):
    def make(self, split: str, batch_size: int, num_workers: int, **kwargs):
        ...  # returns (dataloader, num_classes)
