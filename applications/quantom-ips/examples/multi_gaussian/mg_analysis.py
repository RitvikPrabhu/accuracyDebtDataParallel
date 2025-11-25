import logging
import os
import numpy as np
import torch
from dataclasses import dataclass
import matplotlib.pyplot as plt
from quantom_ips.utils.registration import register_with_hydra

logger = logging.getLogger(__name__)

@dataclass
class MGAnalysisDefaults:
    id: str = "MGAnalysis"
    logdir: str = "${hydra:runtime.output_dir}"
    printout_frequency: int = 1
    frequency: int = 1
    snapshot_frequency: int = 1
    plotstyle: str = "bmh"

@register_with_hydra(
    group="analysis",
    defaults=MGAnalysisDefaults,
    name="mg_analysis",
)
class MGAnalysis:

    def __init__(self, config, mpi_comm, devices, dtype):
        self.config = config
        self.mpi_comm = mpi_comm
        self.devices = devices
        self.dtype = dtype
        self.logdir = config.logdir
        self.frequency = config.frequency
        self.printout_frequency = config.printout_frequency
        self.snapshot_frequency = config.snapshot_frequency
        self.plotstyle = config.plotstyle
        plt.rcParams.update({'font.size':20})
    
        # Get the current rank and the number of ranks:
        self.n_ranks = self.mpi_comm.Get_size()
        self.current_rank = self.mpi_comm.Get_rank()

        # Now create directories where we want to store the results:
        self.model_loc = self.logdir+"/models"
        self.training_loc = self.logdir+"/training"
        os.makedirs(self.model_loc,exist_ok=True)
        os.makedirs(self.training_loc,exist_ok=True)

        # Make sure that everyone is on the same page
        self.mpi_comm.Barrier()
        # Define gradient monitor:
        self.gradient_monitor = None

        # Capture all metrics that are recorded by the trainer:
        self.metrics = {}
    
    def online_monitoring(self, epoch, n_epochs, loss_history, optimizer):
        if loss_history is not None:
            if epoch == 1:
                self.metrics["epochs"] = [epoch]
                self.metrics["snapshot_epochs"] = [epoch]
                # +++++++++++++++++
                for m in loss_history:
                    self.metrics[m] = [loss_history[m]]
                # +++++++++++++++++
            elif epoch % self.frequency == 0:
                self.metrics["epochs"].append(epoch)
                # +++++++++++++++++
                for m in loss_history:
                    self.metrics[m].append(loss_history[m])
                # +++++++++++++++++

        # Print out losses if wanted:
        if (
            self.mpi_comm.Get_rank() == 0
            and self.printout_frequency > 0
            and epoch % self.printout_frequency == 0
        ):
            if self.metrics:
                logger.info(f"Epoch: {epoch}")
                # +++++++++++++++++
                for m in loss_history:
                    if "loss" in m:
                        logger.info(f"{m}: {np.round(loss_history[m],4)}")
                # +++++++++++++++++

        # Write model to file:
        # The trained model:
        if optimizer is not None and (epoch == 1 or epoch % self.snapshot_frequency == 0):
            if epoch > 1:
              self.metrics["snapshot_epochs"].append(epoch)
            epoch_str = str(epoch) + "epochs"
            epoch_str = epoch_str.zfill(6 + len(str(n_epochs)))
            full_model_path = self.model_loc+"/generator_rank"+str(self.current_rank)+"_"+epoch_str+".pt"
            opt_state_dic = optimizer.state_dict()
            torch.save(opt_state_dic,full_model_path)


    def write_to_file(self):
        # Losses:
        if self.metrics:
            # +++++++++++++++++
            for key in self.metrics:
                np.save(
                    self.training_loc
                    + "/"
                    + key
                    + "_rank"
                    + str(self.current_rank)
                    + ".npy",
                    np.array(self.metrics[key]),
                )
            # +++++++++++++++++

        del self.metrics
    

    # Check if a desired metric has been recorded:
    def get_recorded_metrics(self,history,metric_name):
        if metric_name is not None:
          recorded_metrics = []
          for key in history:
            if metric_name in key:
                recorded_metrics.append(key)
          return recorded_metrics
        return list(history.keys())

    def plot_metrics_vs_epochs(self, metrics_dict, y_label, plot_name, metric_name):
        metrics_to_plot = self.get_recorded_metrics(metrics_dict,metric_name)
        if len(metrics_to_plot) > 0:
          with plt.style.context(self.plotstyle):
            fig, ax = plt.subplots(figsize=(12, 8))

            # +++++++++++++++++
            for m in metrics_to_plot:
                ax.plot(self.metrics["epochs"], metrics_dict[m], linewidth=3.0, label=m)
            # +++++++++++++++++

            ax.set_xlabel("Epochs")
            ax.set_ylabel(y_label)
            ax.grid(True)
            ax.legend(fontsize=15)
            fig.tight_layout()
            fig.savefig(plot_name + ".png")
            plt.close(fig)

    def offline_monitoring(self):
        # Gathered metrics:
        gathered_metrics = {}
        if self.metrics:
            # +++++++++++++++++
            for key in self.metrics:
                metrics = self.mpi_comm.gather(self.metrics[key], root=0)
                if self.mpi_comm.Get_rank() == 0:
                        gathered_metrics[key] = np.mean(np.stack(metrics), 0)
            # +++++++++++++++++

            # Take a break:
            self.mpi_comm.Barrier()

            # Create a few monitoring plots:
            if self.mpi_comm.Get_rank() == 0:
                mon_dir = self.logdir + "/monitoring"
                os.makedirs(mon_dir, exist_ok=True)

                if self.metrics:
                    # Plot losses:
                    self.plot_metrics_vs_epochs(
                        gathered_metrics, "Avg. Loss", mon_dir + "/loss_curves", "loss"
                    )
    
    def forward(
        self,
        epoch=None,
        n_epochs=None,
        loss_history=None,
        generator=None,
        is_online=False,
    ):
        if is_online:
            self.online_monitoring(epoch, n_epochs, loss_history, generator)
        else:
            self.offline_monitoring()
            self.write_to_file()
