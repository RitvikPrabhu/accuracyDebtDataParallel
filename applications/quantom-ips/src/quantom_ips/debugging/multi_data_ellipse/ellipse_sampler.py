import torch
import numpy as np

class EllipseSampler:

    def __init__(self,scales=[10,7],torch_dtype=torch.float32,torch_device="cpu"):
        self.torch_dtype = torch_dtype
        self.torch_device = torch_device
        self.scales = scales

    def forward(self,params,n_samples,idx):
        avg_params = torch.mean(params,0)
        a_0 = self.scales[0]*avg_params[0]
        b_0 = self.scales[1]*avg_params[1]

        a_1 = self.scales[1]*avg_params[0]
        b_1 = self.scales[0]*avg_params[1]

        t = torch.rand(size=(n_samples,1),dtype=self.torch_dtype,device=self.torch_device)*2.0*np.pi
        x_0 = a_0*torch.cos(t)
        y_0 = b_0*torch.sin(t)
        x_1 = a_1*torch.cos(t)
        y_1 = b_1*torch.sin(t)

        data_0 = torch.cat([x_0,y_0],dim=1)
        data_1 = torch.cat([x_1,y_1],dim=1)
        all_data = torch.stack([data_0,data_1])

        if idx >=0 and idx <=1:
            return all_data[idx] # was idx
        
        return all_data
