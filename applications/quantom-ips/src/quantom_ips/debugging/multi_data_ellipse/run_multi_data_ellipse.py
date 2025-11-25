from quantom_ips.debugging.multi_data_ellipse.ellipse_trainer import EllipseTrainer

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
import hydra
import torch
import os

@dataclass
class RunMultiDataEllipse:
    id: str = "RunMultiDataEllipse"
    n_epochs: int = 10
    batch_size: int = 500
    n_samples: int = 500
    latent_dim: int = 2
    read_frequency: int = 1
    print_frequency: int = 1
    outer_group_frequency: int = 5
    n_hidden_generator: int = 3
    n_neurons_generator: int = 100
    dropouts_generator: float = 0.0
    lr_generator: float = 1e-5
    n_hidden_discriminator: int = 3
    n_neurons_discriminator: int = 100
    lr_discriminator: float = 1e-4
    dropouts_discriminator: float = 0.0
    data_loc: str = "toy_data_v0"
    master_rank: int = 0
    force_rma_rank_synchronization: bool = False
    group_size: int = 4
    gradient_sync_mode: str = "arar"
    use_weight_grad_only: bool = True
    train_as_ensemble: bool = False
    logdir: str = "results"
    std_loss_scale: float = 0.0
    use_global_scale: bool = False
    ema_alpha: float = -1.0

cs = ConfigStore.instance()
cs.store(name="run_multi_data_ellipse", node=RunMultiDataEllipse)

@hydra.main(version_base=None, config_name="run_multi_data_ellipse")
def run(config) -> None:
    
    trainer = EllipseTrainer(config=config)
    idx = trainer.data_idx
    current_rank = trainer.current_rank

    data = np.stack([
        np.load(config.data_loc+"/data_set_0.npy"),
        np.load(config.data_loc+"/data_set_1.npy") # was 1
    ])
    
    history = trainer.fit(data[idx])
    trainer.comm.Barrier()
    
    logdir = config.logdir
    if current_rank == 0:
        os.makedirs(logdir,exist_ok=True)

    # Write out resulte for each rank:
    current_logdir = logdir +"/results_rank"+str(current_rank)
    os.makedirs(current_logdir,exist_ok=True)

    plt.rcParams.update({'font.size':20})
    fig,ax = plt.subplots(figsize=(12,8))

    ax.plot(history['real_loss'],linewidth=3.0,label='Real')
    ax.plot(history['fake_loss'],linewidth=3.0,label='Fake')
    ax.plot(history['gen_loss'],linewidth=3.0,label='Gen.')
    ax.grid(True)
    ax.legend(fontsize=15)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Normalized Loss')
    fig.savefig(current_logdir+"/loss_curves.png")
    plt.close(fig)

    for key, loss in history.items():
        np.save(current_logdir+"/"+key+".npy",loss)

    noise = torch.normal(mean=0.0,std=1.0,size=(10000,config.latent_dim),dtype=torch.float32,device=trainer.torch_device)
    params = None
    if trainer.ema_generator is not None:
        params = trainer.ema_generator(noise)
    else:
        params = trainer.generator(noise)

    gen_events = trainer.sampler.forward(params,100000,-1).detach().cpu().numpy()
    real_idx = np.random.choice(data[0].shape[0],gen_events[0].shape[0])
    real_data = data[:,real_idx,:]

    fige,axe = plt.subplots(2,2,figsize=(16,8))
    fige.subplots_adjust(wspace=0.35,hspace=0.35)
    fige.suptitle('Real and generated ellipse events')

    axe[0,0].hist(gen_events[0][:,0],100,histtype='step',linewidth=3.0,color='red',label='Gen.')
    axe[0,0].hist(real_data[0][:,0],100,histtype='step',linewidth=3.0,color='black',label='Real')
    axe[0,0].set_xlabel(r'$x_{0}$')
    axe[0,0].grid(True)
    axe[0,0].legend(fontsize=15)

    axe[0,1].hist(gen_events[0][:,1],100,histtype='step',linewidth=3.0,color='red',label='Gen.')
    axe[0,1].hist(real_data[0][:,1],100,histtype='step',linewidth=3.0,color='black',label='Real')
    axe[0,1].set_xlabel(r'$y_{0}$')
    axe[0,1].grid(True)
    axe[0,1].legend(fontsize=15)

    axe[1,0].hist(gen_events[1][:,0],100,histtype='step',linewidth=3.0,color='red',label='Gen.')
    axe[1,0].hist(real_data[1][:,0],100,histtype='step',linewidth=3.0,color='black',label='Real')
    axe[1,0].set_xlabel(r'$x_{1}$')
    axe[1,0].grid(True)
    axe[1,0].legend(fontsize=15)

    axe[1,1].hist(gen_events[1][:,1],100,histtype='step',linewidth=3.0,color='red',label='Gen.')
    axe[1,1].hist(real_data[1][:,1],100,histtype='step',linewidth=3.0,color='black',label='Real')
    axe[1,1].set_xlabel(r'$y_{1}$')
    axe[1,1].grid(True)
    axe[1,1].legend(fontsize=15)
    fige.savefig(current_logdir+"/events.png")
    plt.close(fige)

    np.save(current_logdir+"/gen_events.npy",gen_events)
    np.save(current_logdir+"/real_events.npy",real_data)

    params = params.detach().cpu().numpy()

    figp,axp = plt.subplots(1,2,figsize=(16,8))
    figp.subplots_adjust(wspace=0.35,hspace=0.35)
    figp.suptitle('Predicted Parameters')

    n00,_,_ = axp[0].hist(params[:,0],100)
    axp[0].plot([0.2]*2,[0.0,np.max(n00)],'r--',linewidth=3.0)
    axp[0].grid(True)
    axp[0].set_xlabel(f'$p_{0}$')

    n01,_,_ = axp[1].hist(params[:,1],100)
    axp[1].plot([0.7]*2,[0.0,np.max(n01)],'r--',linewidth=3.0)
    axp[1].grid(True)
    axp[1].set_xlabel(f'$p_{1}$')

    figp.savefig(current_logdir+"/params.png")
    plt.close(figp)

    np.save(current_logdir+"/parameters.npy",params)

if __name__ == "__main__":
    run()





