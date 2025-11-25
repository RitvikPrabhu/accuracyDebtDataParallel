from gen_3body_phase_space import Gen3BodyPhaseSpace
from dalitz_theory import DalitzTheory
import matplotlib.pyplot as plt
from quantom_ips.envs.samplers.loits_2d import LOInverseTransformSampler2D
import hydra
import torch
import numpy as np
import os
import logging
from typing import List, Any, Optional
from omegaconf import MISSING, OmegaConf
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

logger = logging.getLogger(__name__)

@dataclass
class CreateDalitzData:
    n_events: int = 10000
    fractions: list = field(default_factory=lambda: [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    parameters: list = field(default_factory=lambda: [[-1.144,0.219,0.0,0.086,0.0,0.115],[-0.056,-0.049,0.0,-0.063,0.0,0.0],[0.133,0.037,0.0,0.0,0.0,0.0]])
    grid_size: int = 200
    logdir: str = "${hydra:runtime.output_dir}"
    mask_loc: str = ""
    mask_dimension: int = 500
    mask_ranges: list = field(default_factory=lambda: [[-1.5,1.5],[-1.5,1.5]])
    device: str = "cpu"


cs = ConfigStore.instance()
cs.store(name="create_dalitz_data", node=CreateDalitzData)

@hydra.main(version_base=None, config_name="create_dalitz_data")
def run(config) -> None:
    
    logger.info("Set up phase space generator")
    m_omega = 782.660
    m_etaP = 957.780 
    m_eta = 547.862
    m_p = 139.57061
    m_pi0 = 134.9768

    mass_dict = {
      'eta': [m_eta,m_p,m_p,m_pi0],
      'etaP': [m_etaP,m_p,m_p,m_eta],
      'omega': [m_omega,m_p,m_p,m_pi0]
    }

    logdir = config.logdir
    os.makedirs(logdir,exist_ok=True)
    mask_loc = f"{logdir}/{config.mask_loc}"

    phase_space_generator = Gen3BodyPhaseSpace(
        masses=mass_dict,
        mask_dimension=config.mask_dimension,
        mask_ranges=config.mask_ranges,
        output_loc=mask_loc,
    )
    logger.info("Generate phase space mask")
    phase_space_generator.create_phase_space_mask()
    
    logger.info("Load theory and sampler module")
    t_cfg = {
        'id': 'DalitzTheory',
        'grid_size': config.grid_size,
        'mask_dir': mask_loc,
        'eps': 0.0,
        'average': True,
        'ratios': config.fractions
    }
    t_cfg = OmegaConf.create(t_cfg)
    theory = DalitzTheory(config=t_cfg,devices=config.device,dtype=torch.float32)
    
    s_cfg = {
      "average": True,
      "a_min": 0.0,
      "a_max": 1.0,
      "n_interpolations_x": 10,
      "n_interpolations_y": 10,
      "use_threading": False,
      "vmap_randomness": "different",
      "log_space": False,
      "static_weight_tensor": False,
      "id": "LOInverseTransformSampler2D"
    }
    s_cfg = OmegaConf.create(s_cfg)
    sampler = LOInverseTransformSampler2D(config=s_cfg,devices=config.device,dtype=torch.float32)

    true_params = torch.as_tensor(
        config.parameters,
        device=config.device,
        dtype=torch.float32
    )[None,:]
    
    logger.info("Generate events")
    amplitudes = theory.forward(true_params)
    data_sets = sampler.forward(amplitudes[0],amplitudes[1],config.n_events).detach().cpu().numpy()[0]
    filtered_sets = []
    min_evs = 1E99
    
    plt.rcParams.update({'font.size':20})
    fig,ax = plt.subplots(1,2,figsize=(16,8),sharey=True)
    for i in range(data_sets.shape[0]):
        set = data_sets[i]
        
        # Remove nans from the set:
        cond = (np.isnan(set[:,0])) | (np.isnan(set[:,1]))
        set = set[~cond]

        if set.shape[0] < min_evs:
            min_evs = set.shape[0]
        
        filtered_sets.append(set)

        ax[0].hist(set[:,0],100,histtype='step',linewidth=3.0,label=f'Data Set {i}')
        ax[1].hist(set[:,1],100,histtype='step',linewidth=3.0,label=f'Data Set {i}')

    ax[0].grid(True)
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Entries')
    ax[0].legend(fontsize=15) 

    ax[1].grid(True)
    ax[1].set_xlabel('Y')
    ax[1].legend(fontsize=15)

    fig.savefig(f'{logdir}/dalitz_datasets.png')
    plt.close(fig) 

    # Write events to file, but make sure they all have the same number of events:
    for i,set in enumerate(filtered_sets):
        idx = np.random.choice(set.shape[0],min_evs)
        np.save(f'{logdir}/dalitz_data_set{i}.npy',set[idx])
        
if __name__ == "__main__":
    run()


        
        











