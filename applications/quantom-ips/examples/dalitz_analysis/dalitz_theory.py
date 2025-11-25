import torch
import numpy as np
import pickle
from scipy.interpolate import RegularGridInterpolator
import logging
from dataclasses import dataclass, field
from quantom_ips.utils.registration import register_with_hydra

logger = logging.getLogger(__name__)

@dataclass
class DalitzTheoryDefaults:
    id: str = "DalitzTheory"
    mask_dir: str = ""
    grid_size: int = 50
    eps: float = 1e-11
    ratios: list = field(default_factory=lambda: [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    average: bool = True

@register_with_hydra(
    group="environment/theory",
    defaults=DalitzTheoryDefaults,
    name="dalitz_theory",
)
class DalitzTheory:

    def __init__(self,config,devices="cpu",dtype=torch.float32):
        self.config = config
        self.average = config.average
        self.dtype = dtype
        self.devices = devices
        self.ratios = config.ratios
        # Load the mask:
        try:
          with open(f"{config.mask_dir}.pkl",'rb') as f:
             H, xc, yc, omega_flag = pickle.load(f)
        except:
          logger.warning(f"Invalid path: {config.mask_dir}")

        # Get grid ranges and define grid:
        x_in = np.linspace(np.min(xc),np.max(xc),config.grid_size)
        y_in = np.linspace(np.min(yc),np.max(yc),config.grid_size)
        x_g,y_g = np.meshgrid(x_in,y_in)
        
        dG_mask = []
        for i in range(H.shape[0]):
          current_interpolator = RegularGridInterpolator((xc, yc), H[i].T)
          current_mask = current_interpolator((x_g,y_g))
          current_mask = np.where(current_mask>0.0,1.0,config.eps)
          dG_mask.append(current_mask[None,:,:])

        self.n_particles = len(dG_mask)
        self.omega_flag = omega_flag
        
        # Register everything with torch:
        self.grids = [
           torch.as_tensor(x_in,device=devices,dtype=dtype),
           torch.as_tensor(y_in,device=devices,dtype=dtype)
        ]
        self.mask = torch.as_tensor(np.concatenate(dG_mask,0),device=devices,dtype=dtype)
        self.X = torch.as_tensor(x_g,device=devices,dtype=dtype)
        self.Y = torch.as_tensor(y_g,device=devices,dtype=dtype)

    # Dalitz plot parameterization:
    def dalitz_parameterization(self,X,Y,par,flag):
       amplitude = 0.0
       if flag:
          Z = X**2 + Y**2
          Phi = torch.atan2(Y,X)
          amplitude = 1.0 + 2.0*par[0]*Z + 2.0*par[1]*torch.pow(Z,1.5)*torch.sin(3.0*Phi)
       amplitude = 1.0 + par[0]*Y + par[1]*Y*Y + par[2]*X + par[3]*X*X + par[4]*X*Y + par[5]*Y*Y*Y
       max_amp = amplitude.max(dim=0).values
       return amplitude / max_amp

    # Compute the amplitude:
    def compute_amplitude(self,params,X,Y):
       A = []
       for r in range(len(self.ratios)):
          current_A = 0.0
          for i in range(self.n_particles):
            current_A += torch.transpose(
                 self.dalitz_parameterization(X,Y,params[i],self.omega_flag[i]),0,1
            )*self.ratios[r][i]
          A.append(current_A)
            
       A = torch.stack(A)
       return A * self.mask
    
    def forward(self,params):
       A = torch.vmap(lambda p: self.compute_amplitude(p,self.X,self.Y),in_dims=0,randomness="same")(params)

       if self.average:
            A = A.mean(dim=0,keepdim=True)

       # Provide loss as one output argument:
       zero_loss = torch.zeros(
            size=(1,), dtype=self.dtype, device=self.devices, requires_grad=True
       ) 
       return A, self.grids, zero_loss
       



