import torch
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Any, Optional
from mg_pipeline import MGPipeline
from quantom_ips.utils.registration import register_with_hydra

logger = logging.getLogger(__name__)

@dataclass
class MGParserDefaults:
    id: str = "MGParser"
    n_events: int = 1000
    alpha: List[Any] = field(default_factory=lambda: [[1.0,0.0,0.0],[0.0,1.0,0.0]])
    params_1: List[Any] = field(default_factory=lambda: [-1.5,0.2])
    params_2: List[Any] = field(default_factory=lambda: [1.5,0.2])
    seed: int = 1234
    fraction: float = 0.5
    data_idx: Optional[int] = None

@register_with_hydra(
    group="environment/parser",
    defaults=MGParserDefaults,
    name="mg_parser",
)
class MGParser:
    '''
    Parser that produces dataset on the go. 
    '''

    def __init__(self,config,devices="cpu",dtype=torch.float32):
        self.config = config
        self.devices = devices
        self.dtype = dtype
        self.data_idx = config.data_idx
        
        params_1 = torch.as_tensor(config.params_1,dtype=self.dtype,device=self.devices)[None,:]
        params_2 = torch.as_tensor(config.params_2,dtype=self.dtype,device=self.devices)[None,:]
        self.params = torch.stack([params_1,params_2]) 
        self.pipeline = MGPipeline(config.n_events, config.alpha)
        npy_data = self.create_data()
    
        n_local_events = int(config.fraction*config.n_events)
        idx = np.random.choice(npy_data.shape[0],size=n_local_events)
        self.data = torch.as_tensor(npy_data[idx],dtype=self.dtype,device=self.devices)

    def create_data(self):
        # Create data with a fixed seed --> Reproduability:
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = -1
        mps_rng_state = -1
        if torch.cuda.is_available():
              cuda_rng_state = torch.cuda.get_rng_state()
        if torch.backends.mps.is_available():
              mps_rng_state = torch.mps.get_rng_state()

        # Set seed so that we know the data is exactly the same on both ranks:
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
             torch.cuda.manual_seed_all(self.config.seed)
        if torch.backends.mps.is_available():
             torch.mps.manual_seed(self.config.seed)
        
        npy_data = self.pipeline.forward(self.params,self.config.data_idx)
           
        # Undo the fixed seed:
        torch.set_rng_state(cpu_rng_state)
        if torch.cuda.is_available():
              torch.cuda.set_rng_state(cuda_rng_state)
        if torch.backends.mps.is_available():
              torch.mps.set_rng_state(mps_rng_state)
        
        return npy_data.detach().cpu().numpy()

    def forward(self,n_samples):
       if n_samples != self.data.size()[0]:
         idx = torch.randint(self.data.size()[0],size=(n_samples,),device=self.devices)
         return self.data[idx]
       return self.data
