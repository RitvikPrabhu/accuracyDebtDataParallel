import torch
import numpy as np
import hydra
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from quantom_ips.utils.pixelated_densities import DukeAndOwensDensities
from quantom_ips.utils import make
from quantom_ips.envs.theories.duke_and_owens_theory import DukeAndOwensTheory
from quantom_ips.envs.samplers.loits_2d import LOInverseTransformSampler2D
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import logging
import os
import time

logger = logging.getLogger("Duke and Owens Data Generator")

@dataclass
class GenerateDukeAndOwensData:
    n_events: int = 10000
    batch_size: int = 1000 # 1
    grid_size: int = 200
    split_per_batch: bool = False
    data_loc: str = "${hydra:runtime.output_dir}"

cs = ConfigStore.instance()
cs.store(name="generate_duke_and_owens_data", node=GenerateDukeAndOwensData)

@hydra.main(version_base=None, config_name="generate_duke_and_owens_data")
def run(config) -> None:
    # Get the torch device first:
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    if torch.backends.mps.is_available():
        device = "mps"

    logger.info(f"Determine torch device: {device}")

    # Create directory to store all the results:
    logger.info("Create directory for toy data and plots")
    os.makedirs(config.data_loc,exist_ok=True)

    # Get the densities:
    logger.info("Create densities")
    density_generator = DukeAndOwensDensities(devices=device,batch_size=config.batch_size)
    densities = density_generator.get_pixelated_densities(config.grid_size,config.grid_size)

    # Get the theory module:
    logger.info("Load theory and sampler module")
    t_cfg = {
        "a_min": [0.0,0.0],
        "a_max": [1.0,1.0],
        "xsec_min":0.0,
        "xsec_max":1.0,
        "fixed_density_index": -1,
        "average": False,
        "acceptance_epsilon": 1e-11,
    }
    t_cfg = OmegaConf.create(t_cfg)
    theory = make("DukeAndOwensTheory", config=t_cfg, devices=device)

    # Sampler:
    s_cfg = {
        "split_per_batch":config.split_per_batch,
        "parallelize_dim_sampling": False,
        "use_threading": False,
        "vmap_randomness": "different",
        "average": False,
        "a_min": 0.0,
        "a_max": 1.0,
        "log_space": True,
        "n_interpolations_x": 10,
        "n_interpolations_y": 10,
    }
    s_cfg = OmegaConf.create(s_cfg)
    sampler = make(
            "LOInverseTransformSampler2D", config=s_cfg, devices=device
    )

    # Create the data:
    logger.info("Generate toy data")
    t_start = time.time()

    A, grid_axes, _ = theory.forward(densities)
    events = sampler.forward(A, grid_axes, config.n_events).detach().cpu().numpy()
    
    t_end = time.time()

    t_diff = t_end - t_start

    events_proton = np.reshape(events[:,0,:,:],(events[:,0,:,:].shape[0]*events[:,0,:,:].shape[1],events[:,0,:,:].shape[2]))
    events_neutron = np.reshape(events[:,1,:,:],(events[:,1,:,:].shape[0]*events[:,1,:,:].shape[1],events[:,1,:,:].shape[2]))

    print(f"Time needed for generation: {t_diff}")
    print(f"Minima proton: {np.min(events_proton,0)} / Minima neutron: {np.min(events_neutron,0)}")
    print(f"Maxima proton: {np.max(events_proton,0)} / Maxima neutron: {np.max(events_neutron,0)}")


    log_events_proton = np.log(events_proton)
    log_events_neutron = np.log(events_neutron)

    # Plot data:
    logger.info("Visualize toy data")
    plt.rcParams.update({"font.size":20})
    fig,ax = plt.subplots(2,2,figsize=(16,10))
    fig.subplots_adjust(wspace=0.5,hspace=0.5) 
    fig.suptitle("Events generated via Duke & Owens & 2D LOITS")
    
    # 1D Distributions:
    ax[0,0].hist(log_events_proton[:,0],100,histtype='step',linewidth=3.0,color='black',label='Proton')
    ax[0,0].hist(log_events_neutron[:,0],100,histtype='step',linewidth=3.0,color='red',label='Neutron')
    ax[0,0].grid(True)
    ax[0,0].legend(fontsize=15)
    ax[0,0].set_xlabel('Sampled ' + r'$\log(x)$')
    ax[0,0].set_ylabel('Entries')

    ax[0,1].hist(log_events_proton[:,1],100,histtype='step',linewidth=3.0,color='black',label='Proton')
    ax[0,1].hist(log_events_neutron[:,1],100,histtype='step',linewidth=3.0,color='red',label='Neutron')
    ax[0,1].grid(True)
    ax[0,1].legend(fontsize=15)
    ax[0,1].set_xlabel('Sampled ' + r'$\log(Q^2)$')
    ax[0,1].set_ylabel('Entries')
    
    # 2D Distributions:
    ax[1,0].set_title('Proton')
    ax[1,0].hist2d(log_events_proton[:,0],log_events_proton[:,1],100,norm=LogNorm())
    ax[1,0].grid(True)
    ax[1,0].set_xlabel('Sampled ' + r'$\log(x)$')
    ax[1,0].set_ylabel('Sampled ' + r'$\log(Q^2)$')

    ax[1,1].set_title('Neutron')
    ax[1,1].hist2d(log_events_neutron[:,0],log_events_neutron[:,1],100,norm=LogNorm())
    ax[1,1].grid(True)
    ax[1,1].set_xlabel('Sampled ' + r'$\log(x)$')
    ax[1,1].set_ylabel('Sampled ' + r'$\log(Q^2)$')

    fig.savefig(config.data_loc+"/duke_and_owens_data_plots.png")
    plt.close(fig)

    # Write data to file:
    logger.info("Write data to file")
    np.save(config.data_loc+"/events_duke_and_owens_proton.npy",events_proton)
    np.save(config.data_loc+"/events_duke_and_owens_neutron.npy",events_neutron)

if __name__ == "__main__":
    run()
