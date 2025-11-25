import matplotlib.pyplot as plt
import torch
import quantom_ips.utils.plotting as plot_utils
from dataclasses import dataclass
from pathlib import Path

from quantom_ips.utils.registration import register_with_hydra


def detach_tensors(tensor):
    return tensor.detach().cpu()


@dataclass
class BaseAnalysisDefaults:
    id: str = "BaseAnalysis"
    plot_style: str = "bmh"
    logdir: str = "${hydra:runtime.output_dir}"
    batch_size: int = 100
    frequency: int = 500


@register_with_hydra(group="analysis", defaults=BaseAnalysisDefaults, name="base")
class BaseAnalysis:
    def __init__(self, config, devices, dtype):
        self.config = config
        self.devices = devices
        self.dtype = dtype
        self.plot_style = config.plot_style  # bmh
        self.logdir = Path(config.logdir)
        self.batch_size = config.batch_size
        self.frequency = config.frequency

    @torch.no_grad()
    def forward(
        self, optimizer, environment, epoch=None, loss_history=None, force=False
    ):
        """
        Tasks that we'd like to do:
        -- Visualize the real and fake events
        -- Visualize the generator outputs compared to expected (Turn on/off)
        -- TODO: Visualize the discriminator outputs for a range of samples
        -- Visualize the loss history
        """

        if (epoch % self.frequency != 0) and not force:
            return

        fake_params = optimizer.forward(self.batch_size)
        real_params = environment.data_parser.get_pdf()

        fake_params = torch.mean(fake_params, dim=0, keepdim=True)
        _, fake_events = environment.step(fake_params)
        real_events = environment.data_parser.get_samples(fake_events.shape)

        fake_params = detach_tensors(fake_params.squeeze())
        real_params = detach_tensors(real_params)

        fake_events = detach_tensors(fake_events.squeeze())
        real_events = detach_tensors(real_events.squeeze())

        filename = self.logdir.joinpath(f"event_distributions_e{epoch}.pdf")
        plot_utils.plot_2d_event_histogram(real_events, fake_events, save_file=filename)

        filename = self.logdir.joinpath(f"probability_distributions_e{epoch}.pdf")
        plot_utils.plot_2d_pdfs([real_params, fake_params], save_file=filename)

        if epoch is None:
            epoch = -1

        if loss_history is not None:
            self.plot_losses(loss_history)

    def plot_losses(self, loss_history):
        filename = self.logdir.joinpath("loss_history.pdf")
        with plt.style.context(self.plot_style):
            for key, loss in loss_history.items():
                plt.semilogy(loss, label=key)
            plt.legend()
            plt.ylabel("Losses")
            plt.xlabel("Epochs")
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()
            plt.clf()
