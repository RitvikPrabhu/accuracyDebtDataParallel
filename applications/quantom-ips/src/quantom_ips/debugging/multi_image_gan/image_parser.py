import numpy as np
import torch
import logging
from quantom_ips.debugging.multi_image_gan.pipeline import pipeline
from typing import List, Any
from dataclasses import dataclass, field
from quantom_ips.utils.registration import register_with_hydra

logger = logging.getLogger(__name__)

@dataclass
class ImageParserDefaults:
    id: str = "ImageParser"
    logs: List[Any] = field(default_factory=lambda: [-1.0,1.0])
    scales: List[Any] = field(default_factory=lambda: [0.05,0.05])
    alphas: List[Any] = field(default_factory=lambda: [[0.5,0.5],[0.5,0.5]])
    n_points: int = 1000000
    n_bins: int = 64
    n_samples: int = 10
    global_scale: float = 1.0
    relative_offset: float = 0.1
    noise: List[Any] = field(default_factory=lambda: [0.001,0.001])
    x_range: List[Any] = field(default_factory=lambda: [-2.0,2.0])
    y_range: List[Any] = field(default_factory=lambda: [-2.0,2.0])
    seed: int = 123

@register_with_hydra(
    group="environment/parser",
    defaults=ImageParserDefaults,
    name="image_parser",
)
class ImageParser:
    
    def __init__(self,config,devices="cpu",dtype=torch.float32):
        self.logs = config.logs
        self.scales = config.scales
        self.n_bins = config.n_bins
        self.n_points = config.n_points
        self.n_samples = config.n_samples
        self.relative_offset = config.relative_offset
        self.x_range = config.x_range
        self.y_range = config.y_range
        self.global_scale = config.global_scale
        self.alphas = config.alphas
        self.noise = config.noise
        self.seed = config.seed
        self.torch_device = devices
        self.torch_dtype = dtype

        # Set the random seed: 
        # Make sure that we always create the same data samle
        if self.seed >= 0:
           np.random.seed(self.seed)

        self.cpu_rng_state = torch.get_rng_state()
        self.cuda_rng_state = -1
        self.mps_rng_state = -1
        if torch.cuda.is_available():
            self.cuda_rng_state = torch.cuda.get_rng_state()
        if torch.backends.mps.is_available():
           self.mps_rng_state = torch.mps.get_rng_state()

        data, true_pdfs = self.return_data()
        self.data = data
        self.true_pdfs = true_pdfs

    def create(self,data_0,data_1):  
       img_data_0,_,_ = np.histogram2d(data_0[:,0],data_0[:,1],self.n_bins,range=[self.x_range,self.y_range])
       img_data_1,_,_ = np.histogram2d(data_1[:,0],data_1[:,1],self.n_bins,range=[self.x_range,self.y_range])
       
       if self.relative_offset > 0.0:
           img_data_0 = img_data_0 + np.ones_like(img_data_0)*np.max(img_data_0)*self.relative_offset
           img_data_1 = img_data_1 + np.ones_like(img_data_1)*np.max(img_data_1)*self.relative_offset

       img_data_0 = (img_data_0 / np.max(img_data_0)) * self.global_scale
       img_data_1 = (img_data_1 / np.max(img_data_1)) * self.global_scale

       all_densities = [img_data_0,img_data_1] 
       all_data = pipeline(all_densities,self.alphas,False)

       return np.stack(all_data), np.stack(all_densities)
    
    def return_data(self):
        # Set the seed for torch --> We want to make sure that we have "one" data set
        if self.seed >= 0:
          torch.manual_seed(self.seed)
          if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
          if torch.backends.mps.is_available():
            torch.mps.manual_seed(self.seed)

        data_0 = np.random.multivariate_normal([self.logs[0]]*2,[[self.scales[0],0.0],[0.0,self.scales[0]]],size=(self.n_samples,self.n_points))
        data_1 = np.random.multivariate_normal([self.logs[1]]*2,[[self.scales[1],0.0],[0.0,self.scales[1]]],size=(self.n_samples,self.n_points))

        data = []
        for i in range(self.n_samples):
            current_data, pdfs = self.create(data_0[i,:],data_1[i,:])
            data.append(current_data)

        data = torch.as_tensor(np.stack(data),dtype=self.torch_dtype,device=self.torch_device)
        pdfs = torch.as_tensor(pdfs,dtype=self.torch_dtype,device=self.torch_device)
        noisy_data = self.add_noise(data)

        # Undo the fixed seed:
        if self.seed >= 0:
          torch.set_rng_state(self.cpu_rng_state)
          if torch.cuda.is_available():
           torch.cuda.set_rng_state(self.cuda_rng_state)
          if torch.backends.mps.is_available():
           torch.mps.set_rng_state(self.mps_rng_state)

        return noisy_data, pdfs
    
    def get_samples(self,n_samples):
       if n_samples != self.data.size()[0]:
         idx = torch.randint(self.data.size()[0],size=(n_samples,),device=self.torch_device)
         return self.data[idx]
       return self.data
    
    def add_noise(self,data):
        new_data = []
        for i in range(data.size()[1]):
            noise = torch.rand(data[:,i,:,:].size(),dtype=self.torch_dtype,device=self.torch_device)*torch.max(data[:,i,:,:])*self.noise[i]
            new_data.append(data[:,i,:,:] + noise)
        return torch.stack(new_data,dim=1)



