import numpy as np
import logging
import torch
from dataclasses import dataclass, field
from omegaconf import MISSING

from quantom_ips.utils.registration import register_with_hydra

logger = logging.getLogger(__name__)


@dataclass
class ProxyParserDefaults:
    id: str = "ProxyParser"
    path: list = MISSING
    dtype: str = "float32"
    event_axis: int = 0
    pdf_shape: list = field(default_factory=lambda: [10, 10])


@register_with_hydra(
    group="environment/parser", defaults=ProxyParserDefaults, name="proxy_2D"
)
class ProxyParser:

    # Initialize:
    # ***********************************
    def __init__(self, config, devices="cpu", dtype=torch.float32):
        module = "data."
        module_name = "numpy_parser"
        self.full_module_name = module + module_name

        self.data_path = config.path
        self.data_dtype = config.dtype
        self.data_axis = config.event_axis

        self.pdf_shape = config.pdf_shape
        self.devices = devices

        self.dtype = dtype
        self.data = self.load_data()

    # ***********************************

    # Load .npy file(s):
    # ***********************************
    # Load a single file:
    def load_single_file(self, path_to_file):
        try:
            return np.load(path_to_file).astype(self.data_dtype)
        except Exception as e:
            logger.exception(str(e))
            #    f">>> File ({path_to_file}) does not exist! <<<"
        # )

    # -----------------------------

    # Load multiple files which represent the final data:
    def load_data(self):
        try:
            logger.debug(f"Loading Data from {self.data_path}...")
            collected_data = []
            # +++++++++++++++++++++
            for path in self.data_path:
                collected_data.append(self.load_single_file(path))
            # +++++++++++++++++++++
            data = np.concatenate(collected_data, axis=self.data_axis)
            logger.debug("Creating torch Tensor...")
            data = torch.from_numpy(data).to(device=self.devices, dtype=self.dtype)
            return data
        except Exception as e:
            logger.error(
                "Please check the provided data path which must be a list. "
                f"Received {self.data_path}"
            )

            raise e

    # ***********************************

    def get_samples(self, sample_shape):
        # For now, we generate sample_shape[0] * sample_shape[1] samples and reshape
        batch_size, particles, n_samples, dim = sample_shape
        real_sample_idx = np.random.choice(
            self.data.shape[0], size=n_samples * batch_size * particles
        )
        out = self.data[real_sample_idx]

        return out.reshape(sample_shape)

    def get_pdf(self):
        # True parameters:
        # These should be loaded from a file or in the config...
        true_params = np.array([0.67, 0.2, 0.23, 0.67, 0.5], dtype=np.float32)

        # A few functions to compute the pizelized PDFs:
        def A(p, x, y):
            return (
                torch.pow(x, p[0])
                * torch.pow((1.0 - x), p[1])
                * torch.pow(y, p[2])
                * torch.pow((1.0 - y), p[3])
                * (1.0 + p[4] * x * y)
            )

        def get_pixelized_pdf(p):
            p_min = np.array([-0.5, 2.75, 0.0, 3.0, 0.0], dtype=np.float32)
            p_max = np.array([1.0, 4.0, 1.3, 4.5, 1.5], dtype=np.float32)
            p_scaled = torch.as_tensor((p_max - p_min) * p + p_min)

            x = torch.linspace(0.001, 0.999, self.pdf_shape[0])
            y = torch.linspace(0.001, 0.999, self.pdf_shape[1])

            xx, yy = torch.meshgrid(x, y, indexing="ij")
            zz = torch.vmap(lambda par: A(par, xx, yy), in_dims=0)(p_scaled)

            xx = xx.detach().cpu().numpy()
            yy = yy.detach().cpu().numpy()
            zz = zz.detach().cpu().numpy()
            return xx, yy, zz

        # Get the true pdfs:
        _, _, pdf_true = get_pixelized_pdf(true_params.reshape(1, 5))
        return torch.as_tensor(np.squeeze(pdf_true))
