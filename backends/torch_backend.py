import torch
from core.registry import BACKENDS
from .base import BaseBackend

@BACKENDS.register("torch_backend")
class TorchBackend(BaseBackend):
    def __init__(self, device="cuda", compile=False, cudnn_benchmark=True, set_float32_matmul_precision="medium"):
        self.device = device
        self.compile = compile
        self.cudnn_benchmark = cudnn_benchmark
        self.matmul_prec = set_float32_matmul_precision

    def prepare(self, model, amp: bool, dtype: str):
        torch.backends.cudnn.benchmark = self.cudnn_benchmark
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(self.matmul_prec)
        if self.device:
            dev = self.device
            if dev == "cuda" and not torch.cuda.is_available():
                dev = "cpu"
            model = model.to(dev)
        if self.compile and hasattr(torch, "compile"):
            model = torch.compile(model)
        return model
