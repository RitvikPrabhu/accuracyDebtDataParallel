from typing import Protocol

class ModelFactory(Protocol):
    def build(self, num_classes: int):
        ...  # returns a nn.Module or equivalent
