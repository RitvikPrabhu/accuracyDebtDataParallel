import torch
import numpy as np
import hydra
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from quantom_ips.utils.pixelated_densities import ProxyApplication2DDensities
from quantom_ips.utils import make
from quantom_ips.envs.theories.proxy2d_theory import Proxy2DTheory
from quantom_ips.envs.samplers.loits_2d import LOInverseTransformSampler2D
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import logging
import os
import time

logger = logging.getLogger("2D Proxy Data Generator")

@dataclass
class Generate2DProxyData:
    n_events: int = 10000000
    batch_size: int = 1
    grid_size: int = 200
    ratios: list = field(default_factory=lambda: [0.9,0.75])
    split_per_batch: bool = False
    data_loc: str = "${hydra:runtime.output_dir}"

cs = ConfigStore.instance()
cs.store(name="generate_2dproxy_data", node=Generate2DProxyData)

@hydra.main(version_base=None, config_name="generate_2dproxy_data")
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
    density_generator = ProxyApplication2DDensities(devices=device,batch_size=config.batch_size,n_targets=2)
    densities = density_generator.get_pixelated_density(config.grid_size,config.grid_size)

    # Get the theory module:
    logger.info("Load theory and sampler module")
    t_cfg = {
        "a_min": [0.0,0.0],
        "a_max": [1.0,1.0],
        "x_min": 0.001,
        "x_max": 0.999,
        "y_min": 0.001,
        "y_max": 0.999,
        "ratios": config.ratios,
        "average": False,
    }
    t_cfg = OmegaConf.create(t_cfg)
    theory = make("Proxy2DTheory", config=t_cfg, devices=device)

    # Sampler:
    s_cfg = {
        "static_weight_tensor": False,
        "use_threading": False,
        "vmap_randomness": "different",
        "average": False,
        "a_min": 0.0,
        "a_max": 1.0,
        "log_space": False,
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

    events_data0 = np.reshape(events[:,0,:,:],(events[:,0,:,:].shape[0]*events[:,0,:,:].shape[1],events[:,0,:,:].shape[2]))
    events_data1 = np.reshape(events[:,1,:,:],(events[:,1,:,:].shape[0]*events[:,1,:,:].shape[1],events[:,1,:,:].shape[2]))
    events_data2 = None
    if len(config.ratios) > 2:
      events_data2 = np.reshape(events[:,2,:,:],(events[:,2,:,:].shape[0]*events[:,2,:,:].shape[1],events[:,2,:,:].shape[2]))
      
    densities = densities.detach().cpu().numpy()

    print(f"Time needed for generation: {t_diff}")
    print(f"Minimum density0: {np.min(densities[0][0])} / Maximum density0: {np.max(densities[0][0])}")
    print(f"Minimum density1: {np.min(densities[0][1])} / Maximum density1: {np.max(densities[0][1])}")
    print(f"Minima data0: {np.min(events_data0,0)} / Minima data0: {np.min(events_data1,0)}")
    print(f"Maxima data0: {np.max(events_data0,0)} / Maxima data1: {np.max(events_data1,0)}")

    # Plot data:
    logger.info("Visualize toy data")
    plt.rcParams.update({"font.size":20})
    n_ev_col = 2
    if events_data2 is not None:
        n_ev_col = 3
    
    fig,ax = plt.subplots(2,n_ev_col,figsize=(16,10))
    fig.subplots_adjust(wspace=0.5,hspace=0.5) 
    fig.suptitle("Events generated via 2D Proxy & 2D LOITS")
    
    # 1D Distributions:
    ax[0,0].hist(events_data0[:,0],100,histtype='step',linewidth=3.0,color='black',label='Data 0')
    ax[0,0].hist(events_data1[:,0],100,histtype='step',linewidth=3.0,color='red',label='Data 1')
    
    if events_data2 is not None:
        ax[0,0].hist(events_data2[:,0],100,histtype='step',linewidth=3.0,color='blue',label='Data 2')

    ax[0,0].grid(True)
    ax[0,0].legend(fontsize=15)
    ax[0,0].set_xlabel('Sampled x')
    ax[0,0].set_ylabel('Entries')

    ax[0,1].hist(events_data0[:,1],100,histtype='step',linewidth=3.0,color='black',label='Data 0')
    ax[0,1].hist(events_data1[:,1],100,histtype='step',linewidth=3.0,color='red',label='Data 1')

    if events_data2 is not None:
        ax[0,1].hist(events_data2[:,1],100,histtype='step',linewidth=3.0,color='blue',label='Data 2')
    
    ax[0,1].grid(True)
    ax[0,1].legend(fontsize=15)
    ax[0,1].set_xlabel('Sampled y')
    ax[0,1].set_ylabel('Entries')
    
    # 2D Distributions:
    ax[1,0].set_title('Data 0')
    ax[1,0].hist2d(events_data0[:,0],events_data0[:,1],100,norm=LogNorm())
    ax[1,0].grid(True)
    ax[1,0].set_xlabel('Sampled x')
    ax[1,0].set_ylabel('Sampled y')

    ax[1,1].set_title('Data 1')
    ax[1,1].hist2d(events_data1[:,0],events_data1[:,1],100,norm=LogNorm())
    ax[1,1].grid(True)
    ax[1,1].set_xlabel('Sampled x')
    ax[1,1].set_ylabel('Sampled y')

    if events_data2 is not None:
        ax[1,2].set_title('Data 2')
        ax[1,2].hist2d(events_data2[:,0],events_data2[:,1],100,norm=LogNorm())
        ax[1,2].grid(True)
        ax[1,2].set_xlabel('Sampled x')
        ax[1,2].set_ylabel('Sampled y')

    fig.savefig(config.data_loc+"/proxy_data_plots.png")
    plt.close(fig)

    # Write data to file:
    logger.info("Write data to file")
    np.save(config.data_loc+"/events_2dproxy_data0.npy",events_data0)
    np.save(config.data_loc+"/events_2dproxy_data1.npy",events_data1)

    if events_data2 is not None:
        np.save(config.data_loc+"/events_2dproxy_data2.npy",events_data2)

if __name__ == "__main__":
    run()
