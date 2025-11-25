import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from mg_parser import MGParser
from mg_objective import MGObjective
from mg_environment import MGEnvironment
from mg_optimizer import MGOptimizer
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from quantom_ips.utils.registration import register_with_hydra, make
from typing import List, Any
import hydra
from tqdm import tqdm

defaults = [
    {"environment": "mg_environment"},
    {"optimizer": "mg_optimizer"},
    "_self_",
]

@dataclass
class EvalMGDefaults:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    id: str = "EvalMG"
    device: str = "cpu"
    ensemble_loc: str = ""
    logdir: str = "results"
    batch_size: int = 1000
    ensemble_size: int = 1
    n_ranks: int = 4
    seed: int = 1234


cs = ConfigStore.instance()
cs.store(name="eval_mg", node=EvalMGDefaults)

@hydra.main(version_base=None, config_name="eval_mg")
def run(config) -> None:
    # Pretty plotting:
    plt.rcParams.update({'font.size':20})

    # Set the torch seed, so that we can reproduce all results for every analysis
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    if torch.backends.mps.is_available():
            torch.mps.manual_seed(config.seed)

    # Load environment and optimizer:
    env = make(
        id=config.environment.id,
        config=config.environment,
        mpi_comm=None,
        devices=config.device,
        dtype=torch.float32
    )

    opt = make(
        id=config.optimizer.id,
        config=config.optimizer,
        devices=config.device,
        dtype=torch.float32
    )

    # Create noise:
    noise = torch.normal(
        mean=0.0,
        std=1.0,
        size=(config.batch_size, config.optimizer.n_inputs),
        dtype=torch.float32,
        device=config.device
    )

    # Run evaluation:
    eval(
        config.ensemble_loc,
        noise,
        config.ensemble_size,
        config.n_ranks,
        opt,
        env,
        config.logdir
    )


# Gather the predictions from the ensemble:
def gather_ensemble_prediction_single_epoch(result_loc, noise, ensemble_size, n_ranks, opt, current_epoch, n_epochs):
    # Define epoch:
    epoch_str = str(current_epoch) + "epochs"
    epoch_str = epoch_str.zfill(6 + len(str(n_epochs)))

    # Loop over ensemble:
    ensemble_predictions = []
    m_eff = -1
    for m in range(ensemble_size):
        current_predictions = []
        for r in range(n_ranks):
            state_dict_path = f"{result_loc}_v{m}/models/generator_rank{r}_{epoch_str}.pt"
            current_state_dict = torch.load(state_dict_path,map_location=opt.devices)
            opt.model.load_state_dict(current_state_dict)
            current_predictions.append(opt.predict(noise))
            
        current_predictions = torch.mean(torch.stack(current_predictions),0)
        ensemble_predictions.append(current_predictions)
        m_eff += 1
        
    ensemble_predictions = torch.stack(ensemble_predictions)    
    ensemble_mean = torch.mean(ensemble_predictions,0)
    ensemble_err = torch.std(ensemble_predictions,0)

    return ensemble_mean, ensemble_err, epoch_str

# Run pipeline for visualization and analysis
def ana_pipeline(predictions, predictions_err, epoch_str, env, logdir):
    # Get the predicted "distributions":
    _, fake_data = env.step(predictions)
    # Now ask for the real data:
    real_data = env.data_parser.forward(fake_data.size()[0])
    # Detach:
    fake_data = fake_data.detach().cpu().numpy()
    real_data = real_data.detach().cpu().numpy()
    # Data residuals:
    data_residuals = real_data - fake_data

    # Plot real and generated data:
    figd, axd = plt.subplots(2,2,figsize=(16,10),sharey=True)
    figd.subplots_adjust(hspace=0.35,wspace=0.35)

    for j in range(2):
      axd[j,0].hist(real_data[:,j],100,linewidth=3.0,color='black',label='Real',histtype='step')
      axd[j,0].hist(fake_data[:,j],100,linewidth=3.0,color='red',label='Gen.',histtype='step')
      axd[j,0].grid(True)
      axd[j,0].legend(fontsize=15)
      axd[j,0].set_xlabel(f'Data {j+1}')
      axd[j,0].set_ylabel('Entries [a.u.]')

      axd[j,1].hist(data_residuals[:,j],100,linewidth=3.0,color='black',histtype='step')
      axd[j,1].grid(True)
      axd[j,1].set_xlabel(f'Residual Data {j+1}')

    full_path = f'{logdir}/data_comparison_{epoch_str}.png'
    figd.savefig(full_path)
    plt.close(figd)

    # Plot real vs. predicted parameters:
    true_params = env.data_parser.params.detach().cpu().numpy()[:,0,:]
    predictions = predictions.detach().cpu().numpy()
    predictions_err = predictions_err.detach().cpu().numpy()

    figp,axp = plt.subplots(2,2,figsize=(16,10),sharey=True)
    figp.subplots_adjust(wspace=0.35,hspace=0.35)

    for j in range(2):
        current_pred = predictions[j]
        current_truth = true_params[j].ravel()

        counts_mu, _, _ = axp[j,0].hist(current_pred[:,0],100,linewidth=3.0,color='red',label='Prediction',histtype='step')
        axp[j,0].plot([current_truth[0]]*2,[0,np.max(counts_mu)],'k--',linewidth=3.0,label='Truth')
        axp[j,0].grid(True)
        axp[j,0].legend(fontsize=15)
        x_label = r'$\mu_{%i}$' % (j)
        axp[j,0].set_xlabel(x_label)
        axp[j,0].set_ylabel('Entries')

        counts_sigma, _, _ = axp[j,1].hist(current_pred[:,1],100,linewidth=3.0,color='red',label='Prediction',histtype='step')
        axp[j,1].plot([current_truth[1]]*2,[0,np.max(counts_sigma)],'k--',linewidth=3.0,label='Truth')
        axp[j,1].grid(True)
        axp[j,1].legend(fontsize=15)
        x_label = r'$\sigma_{%i}$' % (j)
        axp[j,1].set_xlabel(x_label)
        axp[j,1].set_ylabel('Entries')

    full_path = f'{logdir}/parameter_comparison_{epoch_str}.png'
    figp.savefig(full_path)
    plt.close(figp)

    # Compute average parameter values and errors:
    avg_preds1 = np.mean(predictions[0],0)
    avg_errs1 = np.mean(predictions_err[0],0)
    avg_preds2 = np.mean(predictions[1],0)
    avg_errs2 = np.mean(predictions_err[1],0)

    avg_mu1 = avg_preds1[0]
    avg_mu1_err = avg_errs1[0]
    avg_sigma1 = avg_preds1[1]
    avg_sigma1_err = avg_errs1[1]
    true_mu1 = true_params[0][0]
    true_sigma1 = true_params[0][1]


    avg_mu2 = avg_preds2[0]
    avg_mu2_err = avg_errs2[0]
    avg_sigma2 = avg_preds2[1]
    avg_sigma2_err = avg_errs2[1]
    true_mu2 = true_params[1][0]
    true_sigma2 = true_params[1][1]

    return {
        'mu1': avg_mu1,
        'mu1_err': avg_mu1_err,
        'mu1_true': true_mu1,
        'sigma1': avg_sigma1,
        'sigma1_err': avg_sigma1_err,
        'sigma1_true': true_sigma1,
        'mu2': avg_mu2,
        'mu2_err': avg_mu2_err,
        'mu2_true': true_mu2,
        'sigma2': avg_sigma2,
        'sigma2_err': avg_sigma2_err,
        'sigma2_true': true_sigma2,
    }

# Combine everything:
def eval(result_loc, noise, ensemble_size, n_ranks, opt, env, logdir):
    # Get the epochs:
    epochs_path = f'{result_loc}_v0/training/snapshot_epochs_rank0.npy'
    epochs = np.load(epochs_path)

    # Create helpful directories:
    plot_path = f'{logdir}/plots'
    npy_path = f'{logdir}/npy_data'
    os.makedirs(plot_path,exist_ok=True)
    os.makedirs(npy_path,exist_ok=True)

    # Data gathering:
    mu1 = np.zeros(shape=(epochs.shape[0]))
    mu1_err = np.zeros(shape=(epochs.shape[0]))
    mu1_true =np.zeros(shape=(epochs.shape[0]))
    sigma1 = np.zeros(shape=(epochs.shape[0]))
    sigma1_err = np.zeros(shape=(epochs.shape[0]))
    sigma1_true = np.zeros(shape=(epochs.shape[0]))
    mu2 = np.zeros(shape=(epochs.shape[0]))
    mu2_err = np.zeros(shape=(epochs.shape[0]))
    mu2_true = np.zeros(shape=(epochs.shape[0]))
    sigma2 = np.zeros(shape=(epochs.shape[0]))    
    sigma2_err = np.zeros(shape=(epochs.shape[0]))
    sigma2_true = np.zeros(shape=(epochs.shape[0]))

    # Loop over epochs:
    pbar = tqdm(range(epochs.shape[0]))
    # +++++++++++++++++++
    for i in pbar:
        epoch = epochs[i]
        # Get predictions:
        ensemble_mean, ensemble_err, epoch_str = gather_ensemble_prediction_single_epoch(
            result_loc,
            noise,
            ensemble_size,
            n_ranks,
            opt,
            epoch,
            epochs[-1]
        )

        # Run analysis:
        results = ana_pipeline(
            ensemble_mean,
            ensemble_err,
            epoch_str,
            env,
            plot_path
        )

        # Collect:
        mu1[i] = results['mu1']
        mu1_err[i] = results['mu1_err']
        mu1_true[i] = results['mu1_true']
        sigma1[i] = results['sigma1']
        sigma1_err[i] = results['sigma1_err']
        sigma1_true[i] = results['sigma1_true']
        mu2[i] = results['mu2']
        mu2_err[i] = results['mu2_err']
        mu2_true[i] = results['mu2_true']
        sigma2[i] = results['sigma2']
        sigma2_err[i] = results['sigma2_err']
        sigma2_true[i] = results['sigma2_true']
    # +++++++++++++++++++

    # Make a nice plot:
    fig,ax = plt.subplots(2,2,figsize=(16,10),sharex=True)
    fig.subplots_adjust(wspace=0.35,hspace=0.35)

    # Mu1:
    res_mu1 = mu1_true - mu1
    ax[0,0].errorbar(epochs,res_mu1,mu1_err,fmt='ko',markersize=10,capsize=10,linewidth=0.0,elinewidth=3.0)
    ax[0,0].plot(epochs,[0.0]*epochs.shape[0],'r--',linewidth=3.0)
    ax[0,0].grid(True)
    ax[0,0].set_ylabel('Residuals ' + r'$\mu_0$')
    
    # Sigma1: 
    res_sigma1 = sigma1_true - sigma1
    ax[0,1].errorbar(epochs,res_sigma1,sigma1_err,fmt='ko',markersize=10,capsize=10,linewidth=0.0,elinewidth=3.0)
    ax[0,1].plot(epochs,[0.0]*epochs.shape[0],'r--',linewidth=3.0)
    ax[0,1].grid(True)
    ax[0,1].set_ylabel('Residuals ' + r'$\sigma_0$')

    # Mu2:
    res_mu2 = mu2_true - mu2
    ax[1,0].errorbar(epochs,res_mu2,mu2_err,fmt='ko',markersize=10,capsize=10,linewidth=0.0,elinewidth=3.0)
    ax[1,0].plot(epochs,[0.0]*epochs.shape[0],'r--',linewidth=3.0)
    ax[1,0].grid(True)
    ax[1,0].set_ylabel('Residuals ' + r'$\mu_1$')
    ax[1,0].set_xlabel('Epochs')
    
    # Sigma2: 
    res_sigma2 = sigma2_true - sigma2
    ax[1,1].errorbar(epochs,res_sigma2,sigma2_err,fmt='ko',markersize=10,capsize=10,linewidth=0.0,elinewidth=3.0)
    ax[1,1].plot(epochs,[0.0]*epochs.shape[0],'r--',linewidth=3.0)
    ax[1,1].grid(True)
    ax[1,1].set_ylabel('Residuals ' + r'$\sigma_1$')
    ax[1,1].set_xlabel('Epochs')

    fig.savefig(f'{plot_path}/residuals.png')
    plt.close(fig)

    np.save(f'{npy_path}/mu1.npy',mu1)
    np.save(f'{npy_path}/mu1_err.npy',mu1_err)
    np.save(f'{npy_path}/mu1_true.npy',mu1_true)
    np.save(f'{npy_path}/sigma1.npy',sigma1)
    np.save(f'{npy_path}/sigma1_err.npy',sigma1_err)
    np.save(f'{npy_path}/sigma1_true.npy',sigma1_true)

    np.save(f'{npy_path}/mu2.npy',mu2)
    np.save(f'{npy_path}/mu2_err.npy',mu2_err)
    np.save(f'{npy_path}/mu2_true.npy',mu2_true)
    np.save(f'{npy_path}/sigma2.npy',sigma2)
    np.save(f'{npy_path}/sigma2_err.npy',sigma2_err)
    np.save(f'{npy_path}/sigma2_true.npy',sigma2_true)

if __name__ == "__main__":
    run()





    







    





  