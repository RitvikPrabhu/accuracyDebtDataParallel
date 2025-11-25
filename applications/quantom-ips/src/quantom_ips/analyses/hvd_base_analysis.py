import logging
import os
import numpy as np
import torch
import horovod.torch as hvd
from dataclasses import dataclass
import matplotlib.pyplot as plt

from quantom_ips.utils.monitors.torch_gradient_snapshot import TorchGradientSnapshot
from quantom_ips.utils.registration import register_with_hydra

logger = logging.getLogger(__name__)


@dataclass
class HVDBaseAnalysisDefaults:
    id: str = "HVDBaseAnalysis"
    logdir: str = "${hydra:runtime.output_dir}"
    plot_style: str = "bmh"
    model_snapshot_frequency: int = 2
    gradient_snapshot_frequency: int = 2
    printout_frequency: int = 1
    frequency: int = 1


@register_with_hydra(group="analysis", defaults=HVDBaseAnalysisDefaults, name="hvd_base_analysis")
class HVDBaseAnalysis:
    """
    Horovod-friendly analogue of DistributedBaseAnalysis.
    """

    def __init__(self, config, devices, dtype):
        if not hvd.is_initialized():
            raise RuntimeError("HVDBaseAnalysis requires horovod.init().")

        self.config = config
        self.devices = devices
        self.dtype = dtype
        self.logdir = config.logdir
        self.frequency = config.frequency
        self.printout_frequency = config.printout_frequency
        self.model_snapshot_frequency = self.config.model_snapshot_frequency
        self.gradient_snapshot_frequency = self.config.gradient_snapshot_frequency
        self.plot_style = self.config.plot_style
        plt.rcParams.update({"font.size": 20})

        self.rank = hvd.rank()
        self.world_size = hvd.size()

        if self.rank == 0:
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.logdir + "/model_snapshots_rank0", exist_ok=True)
            os.makedirs(self.logdir + "/model_performance_rank0", exist_ok=True)

        self.model_snapshot_folder = self.logdir + f"/model_snapshots_rank{self.rank}"
        self.model_perforance_folder = self.logdir + f"/model_performance_rank{self.rank}"

        self.losses = {}
        self.grads = {}
        self.gradient_monitor = None

    def online_monitoring(self, optimizer, epoch, n_epochs, loss_history, data_set_idx):
        model_for_snapshot = optimizer
        if not hasattr(optimizer, "state_dict") and hasattr(optimizer, "model"):
            model_for_snapshot = optimizer.model

        if epoch == 1 and model_for_snapshot is not None:
            self.gradient_monitor = TorchGradientSnapshot(model=model_for_snapshot)

        if loss_history is not None:
            if epoch == 1:
                self.losses["epochs"] = [epoch]
                for m in loss_history:
                    self.losses[m] = [loss_history[m]]
            elif epoch % self.frequency == 0:
                self.losses["epochs"].append(epoch)
                for m in loss_history:
                    self.losses[m].append(loss_history[m])

        if self.rank == 0 and self.printout_frequency > 0 and epoch % self.printout_frequency == 0:
            if loss_history:
                logger.info(f"Epoch: {epoch}")
                for m in loss_history:
                    if "loss" in m:
                        logger.info(f"{m}: {np.round(loss_history[m], 4)}")

        if (
            self.gradient_snapshot_frequency > 0
            and self.gradient_monitor is not None
            and model_for_snapshot is not None
        ):
            if epoch == 2:
                self.grads["gradient_snapshot_epochs"] = [epoch]
                grad_dict = self.gradient_monitor.take_snapshot()
                for m in grad_dict:
                    self.grads[m] = [grad_dict[m]]
            elif epoch % self.gradient_snapshot_frequency == 0:
                self.grads["gradient_snapshot_epochs"].append(epoch)
                grad_dict = self.gradient_monitor.take_snapshot()
                for m in grad_dict:
                    self.grads[m].append(grad_dict[m])

        if epoch == 1 or epoch % self.model_snapshot_frequency == 0:
            if self.rank == 0 and model_for_snapshot is not None and hasattr(
                model_for_snapshot, "state_dict"
            ):
                self.losses.setdefault("model_snapshot_epochs", []).append(epoch)
                epoch_str = str(epoch) + "epochs"
                if n_epochs and n_epochs > 0:
                    epoch_str = epoch_str.zfill(6 + len(str(n_epochs)))
                model_loc = f"{self.model_snapshot_folder}/optimizer_rank{self.rank}_{epoch_str}.pt"
                current_state_dict = model_for_snapshot.state_dict()
                if data_set_idx is not None and data_set_idx >= 0:
                    current_state_dict["data_set_idx"] = data_set_idx
                torch.save(current_state_dict, model_loc)

    def write_recordings_to_file(self):
        if self.rank != 0:
            return
        if self.losses:
            for key in self.losses:
                np.save(
                    f"{self.model_perforance_folder}/{key}_rank{self.rank}.npy",
                    np.array(self.losses[key]),
                )
        if self.grads:
            for key in self.grads:
                np.save(
                    f"{self.model_perforance_folder}/{key}_rank{self.rank}.npy",
                    np.stack(self.grads[key], 0),
                )
        del self.losses, self.grads

    def forward(
        self,
        optimizer=None,
        epoch=None,
        n_epochs=None,
        loss_history=None,
        is_online=False,
        force=False,
        data_set_idx=None,
    ):
        if is_online:
            self.online_monitoring(optimizer, epoch, n_epochs, loss_history, data_set_idx)
        if force:
            self.write_recordings_to_file()
