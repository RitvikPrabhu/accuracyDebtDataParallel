import torch
import numpy as np
from dataclasses import dataclass
import os
from quantom_ips.utils.monitors.torch_gradient_snapshot import TorchGradientSnapshot
from quantom_ips.utils.registration import register_with_hydra
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def detach_tensors(tensor):
    return tensor.detach().cpu()


@dataclass
class DistributedBaseAnalysisDefaults:
    id: str = "DistributedBaseAnalysis"
    logdir: str = "${hydra:runtime.output_dir}"
    plot_style: str = "bmh"
    model_snapshot_frequency: int = 2
    gradient_snapshot_frequency: int = 2
    printout_frequency: int = 1
    frequency: int = 1


@register_with_hydra(
    group="analysis",
    defaults=DistributedBaseAnalysisDefaults,
    name="distributed_base_analysis",
)
class DistributedBaseAnalysis:

    # Initialize:
    # **********************************************
    def __init__(self, config, mpi_comm, devices, dtype):
        self.config = config
        self.mpi_comm = mpi_comm
        self.devices = devices
        self.dtype = dtype
        self.logdir = config.logdir
        self.frequency = config.frequency
        self.printout_frequency = config.printout_frequency
        self.model_snapshot_frequency = self.config.model_snapshot_frequency
        self.gradient_snapshot_frequency = self.config.gradient_snapshot_frequency
        self.plot_style = self.config.plot_style
        plt.rcParams.update({"font.size": 20})

        # Get the current rank and the number of ranks:
        self.n_ranks = self.mpi_comm.Get_size()
        self.current_rank = self.mpi_comm.Get_rank()

        # Now create directories where we want to store the results (each rank ensures its own).
        os.makedirs(self.logdir, exist_ok=True)
        os.makedirs(self.logdir + "/model_snapshots_rank" + str(self.current_rank), exist_ok=True)
        os.makedirs(self.logdir + "/model_performance_rank" + str(self.current_rank), exist_ok=True)

        self.model_snapshot_folder = (
            self.logdir + "/model_snapshots_rank" + str(self.current_rank)
        )
        self.model_perforance_folder = (
            self.logdir + "/model_performance_rank" + str(self.current_rank)
        )
        # Make sure that everyone is on the same page
        self.mpi_comm.Barrier()
        # Define gradient monitor:
        self.gradient_monitor = None

        # Capture all metrics that are recorded by the trainer:
        self.losses = {}
        self.grads = {}

    # **********************************************

    # Online monitoring, i.e. recording losses and such during training:
    # **********************************************
    def online_monitoring(self, optimizer, epoch, n_epochs, loss_history, data_set_idx):
        # Register the gradient monitor:
        if epoch == 1:
            self.gradient_monitor = TorchGradientSnapshot(model=optimizer)

        # Record losses, cuda utilization and log[SAD] scores:
        if loss_history is not None:
            if epoch == 1:
                self.losses["epochs"] = [epoch]
                # +++++++++++++++++
                for m in loss_history:
                    self.losses[m] = [loss_history[m]]
                # +++++++++++++++++
            elif epoch % self.frequency == 0:
                self.losses["epochs"].append(epoch)
                # +++++++++++++++++
                for m in loss_history:
                    self.losses[m].append(loss_history[m])
                # +++++++++++++++++

        # Print out losses if wanted:
        if (
            self.mpi_comm.Get_rank() == 0
            and self.printout_frequency > 0
            and epoch % self.printout_frequency == 0
        ):
            if self.losses:
                logger.info(f"Epoch: {epoch}")
                # +++++++++++++++++
                for m in loss_history:
                    if "loss" in m:
                        logger.info(f"{m}: {np.round(loss_history[m],4)}")
                # +++++++++++++++++

        # Track the gradients:
        if self.gradient_snapshot_frequency > 0:
            if epoch == 2:
                self.grads["gradient_snapshot_epochs"] = [epoch]
                grad_dict = self.gradient_monitor.take_snapshot()
                # +++++++++++++++++
                for m in grad_dict:
                    self.grads[m] = [grad_dict[m]]
                # +++++++++++++++++
            elif epoch % self.gradient_snapshot_frequency == 0:
                self.grads["gradient_snapshot_epochs"].append(epoch)
                grad_dict = self.gradient_monitor.take_snapshot()
                # +++++++++++++++++
                for m in grad_dict:
                    self.grads[m].append(grad_dict[m])
                # +++++++++++++++++

        # Write the optimizer to file:
        if epoch == 1 or epoch % self.model_snapshot_frequency == 0:
            if epoch == 1:
                self.losses["model_snapshot_epochs"] = [epoch]
            else:
                self.losses["model_snapshot_epochs"].append(epoch)

            epoch_str = str(epoch) + "epochs"
            if n_epochs > 0 and n_epochs is not None:
                epoch_str = epoch_str.zfill(6 + len(str(n_epochs)))
            model_loc = (
                self.model_snapshot_folder
                + "/optimizer_rank"
                + str(self.current_rank)
                + "_"
                + epoch_str
                + ".pt"
            )
            current_state_dict = optimizer.state_dict()
            if data_set_idx is not None and data_set_idx >= 0:
                current_state_dict['data_set_idx'] = data_set_idx
            torch.save(current_state_dict, model_loc)

    # **********************************************

    # Write the recordings to file:
    # **********************************************
    def write_recordings_to_file(self):
        # Losses:
        if self.losses:
            # +++++++++++++++++
            for key in self.losses:
                np.save(
                    self.model_perforance_folder
                    + "/"
                    + key
                    + "_rank"
                    + str(self.current_rank)
                    + ".npy",
                    np.array(self.losses[key]),
                )
            # +++++++++++++++++

        # Gradients:
        if self.grads:
            # +++++++++++++++++
            for key in self.grads:
                np.save(
                    self.model_perforance_folder
                    + "/"
                    + key
                    + "_rank"
                    + str(self.current_rank)
                    + ".npy",
                    np.stack(self.grads[key], 0),
                )
            # +++++++++++++++++

        # Delete what we dont need:
        del self.losses, self.grads

    # **********************************************

    # Write a small offline analysis --> Get basic monitoring plots:
    # **********************************************
    # Check if a desired metric has been recorded:
    def get_recorded_metrics(self,history,metric_name):
        if metric_name is not None:
          recorded_metrics = []
          for key in history:
            if metric_name in key:
                recorded_metrics.append(key)
          return recorded_metrics
        return list(history.keys())
    
    # ---------------------------------

    # Helper function to plot metrics:
    def plot_metrics_vs_epochs(self, metrics_dict, y_label, plot_name, metric_name):
        metrics_to_plot = self.get_recorded_metrics(metrics_dict,metric_name)
        if len(metrics_to_plot) > 0:
          with plt.style.context(self.plot_style):
            fig, ax = plt.subplots(figsize=(12, 8))

            # +++++++++++++++++
            for m in metrics_to_plot:
                ax.plot(self.losses["epochs"], metrics_dict[m], linewidth=3.0, label=m)
            # +++++++++++++++++

            ax.set_xlabel("Epochs")
            ax.set_ylabel(y_label)
            ax.grid(True)
            ax.legend(fontsize=15)
            fig.tight_layout()
            fig.savefig(plot_name + ".png")
            plt.close(fig)

    # -------------------------------------

    def offline_monitoring(self):
        # Gathered metrics:
        gathered_losses = {}
        gathered_scores = {}
        gathered_gpu_mem = {}
        gathered_gpu_util = {}
        gathered_cgs = {}
        if self.losses:
            # +++++++++++++++++
            for key in self.losses:
                metrics = self.mpi_comm.gather(self.losses[key], root=0)
                if self.mpi_comm.Get_rank() == 0:
                    avg_stacked_metric = np.mean(np.stack(metrics), 0)
                    if "loss" in key:
                        gathered_losses[key] = avg_stacked_metric
                    if "score" in key:
                        gathered_scores[key] = avg_stacked_metric
                    if "gpu_mem_fraction" in key:
                        gathered_gpu_mem[key] = avg_stacked_metric
                    if "gpu_utilization" in key:
                        gathered_gpu_util[key] = avg_stacked_metric
                    if "cg_metric" in key:
                        gathered_cgs[key] = avg_stacked_metric
            # +++++++++++++++++

            # Take a break:
            self.mpi_comm.Barrier()

            # Create a few monitoring plots:
            if self.mpi_comm.Get_rank() == 0:
                mon_dir = self.logdir + "/monitoring"
                os.makedirs(mon_dir, exist_ok=True)

                if self.losses:
                    # Plot losses:
                    self.plot_metrics_vs_epochs(
                        gathered_losses, "Avg. Loss", mon_dir + "/loss_curves", "loss"
                    )
                    # GPU memory:
                    self.plot_metrics_vs_epochs(
                        gathered_gpu_mem,
                        "Avg. GPU Memory Fraction",
                        mon_dir + "/gpu_mem_usage",
                        "gpu_mem_fraction"
                    )
                    # GPU utilization:
                    self.plot_metrics_vs_epochs(
                        gathered_gpu_util,
                        "Avg. GPU Utilization",
                        mon_dir + "/gpu_utilization",
                        "gpu_utilization"
                    )
                    # Other scores:
                    self.plot_metrics_vs_epochs(
                        gathered_scores, "Avg. Score", mon_dir + "/scores", "log_sad_score"
                    )
                    # And plot conflicting gradient metric, just to be sure:
                    self.plot_metrics_vs_epochs(
                        gathered_cgs,'Avg. CG pos. Metric', mon_dir+"/cg_metric_pos","cg_metric_pos"
                    )
                    self.plot_metrics_vs_epochs(
                        gathered_cgs,'Avg. CG neg. Metric', mon_dir+"/cg_metric_neg","cg_metric_neg"
                    )

    # **********************************************

    # Define forward pass:
    # **********************************************
    def forward(
        self,
        optimizer=None,
        epoch=None,
        n_epochs=None,
        loss_history=None,
        is_online=False,
        data_set_idx=None
    ):
        if is_online:
            self.online_monitoring(optimizer, epoch, n_epochs, loss_history, data_set_idx)
        else:
            self.offline_monitoring()
            self.write_recordings_to_file()

    # **********************************************
