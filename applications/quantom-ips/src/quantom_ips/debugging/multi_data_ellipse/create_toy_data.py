from quantom_ips.debugging.multi_data_ellipse.ellipse_sampler import EllipseSampler
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

data_loc = "toy_data_v0"
os.makedirs(data_loc,exist_ok=True)

plt.rcParams.update({'font.size':20})
torch_device = "mps"
n_events = 10000000

p_0 = 0.2
p_1 = 0.7
predictions = torch.as_tensor([p_0,p_1],device=torch_device)[None,:]
sampler = EllipseSampler(torch_device=torch_device)
events = sampler.forward(predictions,n_events,-1).detach().cpu().numpy()


fig,ax = plt.subplots(2,2,figsize=(16,10),sharex=True)
fig.subplots_adjust(wspace=0.35)
fig.suptitle("Ellipse Toy Data")


ax[0,0].hist(events[0,:,0],100)
ax[0,0].grid(True)
ax[0,0].set_xlabel(f'$x_0$')

ax[0,1].hist(events[0,:,1],100)
ax[0,1].grid(True)
ax[0,1].set_xlabel(f'$y_0$')

ax[1,0].hist(events[1,:,0],100)
ax[1,0].grid(True)
ax[1,0].set_xlabel(f'$x_1$')

ax[1,1].hist(events[1,:,1],100)
ax[1,1].grid(True)
ax[1,1].set_xlabel(f'$y_1$')

fig.savefig(data_loc+"/plots.png")
plt.close(fig)

np.save(data_loc+"/data_set_0.npy",events[0])
np.save(data_loc+"/data_set_1.npy",events[1])


