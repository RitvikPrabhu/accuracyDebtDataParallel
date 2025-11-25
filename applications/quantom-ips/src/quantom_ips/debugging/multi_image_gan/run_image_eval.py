import numpy as np
from image_environment import ImageEnvironment
from image_objective import ImageObjective
from image_parser import ImageParser
from image_optimizer import ImageOptimizer
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from quantom_ips.utils.registration import register_with_hydra, make
from quantom_ips.utils.log_sad_score import compute_logSAD
from torch_model_components import ConcreteDropout
from typing import List, Any
import hydra
import torch
import os
import matplotlib.pyplot as plt

defaults = [
    {"environment": "image_environment"},
    {"optimizer": "image_optimizer"},
    "_self_",
]

@dataclass
class RunImageEvalDefaults:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    id: str = "RunImageEval"
    device: str = "cpu"
    result_loc: str = ""
    logdir: str = "results"
    v_min: float = 0.0
    v_max: float = 1.0
    v_min_ra: float = 0.8
    v_max_ra: float = 1.2
    v_min_re: float = -1.0
    v_max_re: float = 1.0
    plot_cmap: str = 'twilight'
    aoi_alpha: float = 0.3
    ensemble_size: int = 2
    n_repetitions: int = 5
    single_idx: int = 0
    n_ranks: int = 2
    seed: int = 123
    batch_size: int = 10
    mode: str = "ensemble"

cs = ConfigStore.instance()
cs.store(name="run_image_eval", node=RunImageEvalDefaults)

@hydra.main(version_base=None, config_name="run_image_eval")
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

    # Now collect the predictions from the ensemble:
    with torch.no_grad():
      predictions = None
      if config.mode.lower() == "ensemble":
        predictions = gather_ensemble_prediction(
          result_loc=config.result_loc,
          ensemble_size=config.ensemble_size,
          n_ranks=config.n_ranks,
          batch_size=config.batch_size,
          opt=opt,
          env=env,
          single_idx=-1
        )
      
      if config.mode.lower() == "single":
        predictions = gather_ensemble_prediction(
          result_loc=config.result_loc,
          ensemble_size=config.ensemble_size,
          n_ranks=config.n_ranks,
          batch_size=config.batch_size,
          opt=opt,
          env=env,
          single_idx=config.single_idx
        )

      if config.mode.lower() == "mc":
        predictions = gather_mc_prediction(
          result_loc=config.result_loc,
          n_repetitions=config.n_repetitions,
          n_ranks=config.n_ranks,
          batch_size=config.batch_size,
          opt=opt,
          env=env
        )
      
      eval_logdir = config.logdir
      os.makedirs(eval_logdir,exist_ok=True)
      viz_pipeline(
          predictions=predictions['mean'],
          predictions_err=predictions['err'],
          residuals=predictions['residual_mean'],
          residuals_err=predictions['residual_err'],
          true_pdfs=predictions['truth'],
          env=env,
          logdir=eval_logdir,
          plot_cmap=config.plot_cmap,
          v_max=config.v_max,
          v_min=config.v_min,
          v_max_ra=config.v_max_ra,
          v_min_ra=config.v_min_ra,
          v_max_re=config.v_max_re,
          v_min_re=config.v_min_re,
          aoi_alpha=config.aoi_alpha

      )

# Run pipeline for visualization
def viz_pipeline(predictions,predictions_err,residuals,residuals_err,true_pdfs,env,logdir,plot_cmap,v_min,v_max,v_min_ra,v_max_ra,v_min_re,v_max_re,aoi_alpha):
    # Create directory to store .npy data:
    npy_logdir = logdir+"/npy_data"
    os.makedirs(npy_logdir,exist_ok=True)

    # Get the predicted "distributions":
    _, fake_data = env.step(predictions)
    # Now ask for the real data:
    real_data = env.data_parser.get_samples(fake_data.size()[0])
    # Detach:
    fake_data = np.mean(fake_data.detach().cpu().numpy(),0)
    real_data = np.mean(real_data.detach().cpu().numpy(),0)

    # Retreive true densities:
    true_pdfs = true_pdfs.detach().cpu().numpy()
    # Get mean of predicted densities:
    avg_predictions = np.mean(predictions.detach().cpu().numpy(),0)
    avg_residuals = np.mean(residuals.detach().cpu().numpy(),0)
    
    # Monitor uncertainty, if available:
    if predictions_err is not None:
        predictions_err = np.mean(predictions_err.detach().cpu().numpy(),0)
        residuals_err = np.mean(residuals_err.detach().cpu().numpy(),0)
    for i in range(2):
        figb, axb = plt.subplots(1,2,figsize=(16,10),sharex=True,sharey=True)

        axb[0].set_title(f'Gen. Data B{i}')
        axb[0].pcolormesh(fake_data[i],vmin=v_min,vmax=v_max,cmap=plot_cmap)
        axb[0].grid(True)
        axb[0].set_xticks([])
        axb[0].set_yticks([])

        axb[1].set_title(f'Real Data R{i}')
        img = axb[1].pcolormesh(real_data[i],vmin=v_min,vmax=v_max,cmap=plot_cmap)
        axb[1].grid(True)
        axb[1].set_xticks([])
        axb[1].set_yticks([])

        figb.colorbar(img, ax=axb.ravel().tolist(), orientation='vertical')
        figb.savefig(f"{logdir}/plot_data_BR{i}.png")
        plt.close(figb)

        # Store the data:
        np.save(f'{npy_logdir}/gen_data_B{i}.npy',fake_data[i])
        np.save(f'{npy_logdir}/real_data_R{i}.npy',real_data[i])


        # Compute ratio between generated and true:
        ratio = avg_predictions[i] / true_pdfs[i]
        # Get the log[SAD] score:
        current_log_sad = compute_logSAD(true_pdfs[i],avg_predictions[i])
        # And plot everything:
        figa, axa = plt.subplots(2,2,figsize=(18,10),sharex=True,sharey=True)
        figa.suptitle(f'log[SAD] = {np.round(current_log_sad,2)}')
        
        axa[0,0].set_title(f'Gen. Density A{i}')
        axa[0,0].pcolormesh(avg_predictions[i],cmap=plot_cmap,vmin=v_min,vmax=v_max)
        axa[0,0].set_xticks([])
        axa[0,0].set_yticks([])
        axa[0,0].grid(True)
       
        axa[0,1].set_title(f'Real Density A{i}')
        imga= axa[0,1].pcolormesh(true_pdfs[i],cmap=plot_cmap,vmin=v_min,vmax=v_max)
        axa[0,1].set_xticks([])
        axa[0,1].set_yticks([])
        axa[0,1].grid(True)

        axa[1,0].set_title(f'Gen. / Real')
        imgra = axa[1,0].pcolormesh(ratio,vmin=v_min_ra,vmax=v_max_ra,cmap=plot_cmap)
        axa[1,0].set_xticks([])
        axa[1,0].set_yticks([])
        axa[1,0].grid(True)

        axa[1,1].set_title(f'Real - Gen.')
        imgre = axa[1,1].pcolormesh(avg_residuals[i],vmin=v_min_re,vmax=v_max_re,cmap=plot_cmap)
        axa[1,1].set_xticks([])
        axa[1,1].set_yticks([])
        axa[1,1].grid(True)
        
        figa.colorbar(imga, ax=axa[0,1], orientation='vertical')
        figa.colorbar(imgra, ax=axa[1,0], orientation='vertical')
        figa.colorbar(imgre, ax=axa[1,1], orientation='vertical')
        figa.savefig(f'{logdir}/plot_density_A{i}.png')
        plt.close(figa)

        # Get 1D projections of ratio and residual:
        ratio_flat = ratio.flatten()
        residual_flat = avg_residuals[i].flatten()
        
        figr,axr = plt.subplots(1,2,figsize=(16,8),sharey=True)

        axr[0].hist(ratio_flat,50,histtype='step',linewidth=3.0)
        axr[0].grid(True)
        axr[0].set_xlabel('Ratio Gen. / Real')
        axr[0].set_ylabel('Entries [a.u.]')

        axr[1].hist(residual_flat,50,histtype='step',linewidth=3.0)
        axr[1].grid(True)
        axr[1].set_xlabel('Residual Real - Gen.')
        
        figr.savefig(f'{logdir}/plot_flat_ratio_residual_A{i}.png')
        plt.close(figr)
        
        # Store the densitties:
        np.save(f'{npy_logdir}/true_pdf_A{i}.npy',true_pdfs[i])
        np.save(f'{npy_logdir}/gen_pdf_A{i}.npy',avg_predictions[i])
        np.save(f'{npy_logdir}/ratio_flat_A{i}.npy',ratio_flat)
        np.save(f'{npy_logdir}/residual_flat_A{i}.npy',residual_flat)
        np.save(f'{npy_logdir}/logSAD_A{i}.npy',current_log_sad)

        # Plot uncertainties, if exist:
        if predictions_err is not None:
            fige,axe = plt.subplots(2,2,figsize=(18,10))
            fige.subplots_adjust(wspace=0.5,hspace=0.5)
            
            # Absolute uncertainties:
            axe[0,0].set_title(f'Uncertainy Density A{i}')
            axe[0,0].pcolormesh(predictions_err[i],cmap=plot_cmap,vmin=v_min,vmax=v_max)
            axe[0,0].set_xticks([])
            axe[0,0].set_yticks([])
            axe[0,0].grid(True)
            fige.colorbar(imga, ax=axe[0,0], orientation='vertical')

            # Now we produce 1D projections:
            err_flat = predictions_err[i].flatten()
            aoi_flat = true_pdfs[i].flatten()

            axe[0,1].plot(aoi_flat,'k.',label='Area of Interest',alpha=aoi_alpha)
            axe[0,1].plot(err_flat,'r.',label=f'Density A{i}',markersize=7)
            axe[0,1].grid(True)
            axe[0,1].set_xlabel('Flattened Bins [a.u.]')
            axe[0,1].set_ylabel('Uncertainty')
            axe[0,1].legend(fontsize=15)

            # Uncertainties on residuals:
            axe[1,0].set_title(f'Uncertainy Residual A{i}')
            axe[1,0].pcolormesh(residuals_err[i],cmap=plot_cmap,vmin=v_min,vmax=v_max)
            axe[1,0].set_xticks([])
            axe[1,0].set_yticks([])
            axe[1,0].grid(True)
            fige.colorbar(imga, ax=axe[1,0], orientation='vertical')

            residual_err_flat = residuals_err[i].flatten()

            axe[1,1].plot(aoi_flat,'k.',label='Area of Interest',alpha=aoi_alpha)
            axe[1,1].plot(residual_err_flat,'r.',label=f'Density A{i}',markersize=7)
            axe[1,1].grid(True)
            axe[1,1].set_xlabel('Flattened Bins [a.u.]')
            axe[1,1].set_ylabel('Residual Uncertainty')
            axe[1,1].legend(fontsize=15)

            fige.savefig(f'{logdir}/plot_uncertainty_A{i}.png')
            plt.close(fige)

            # Also write uncertainties to file, in case we want to plot them later:
            np.save(f'{npy_logdir}/uncertainty_A{i}.npy',predictions_err[i])
            np.save(f'{npy_logdir}/uncertainty_flat_A{i}.npy',err_flat)
            np.save(f'{npy_logdir}/residual_uncertainty_flat_A{i}.npy',residual_err_flat)
            np.save(f'{npy_logdir}/area_of_interest_flat_A{i}.npy',aoi_flat)


# Gather the predictions from the ensemble:
def gather_ensemble_prediction(result_loc,ensemble_size,n_ranks,batch_size,opt,env,single_idx):
    # Get the true predictions:
    true_predictions = env.data_parser.true_pdfs
    # Create noise:
    noise = torch.normal(mean=0.0,std=1.0,size=(batch_size,opt.config.nz,1,1),device=opt.devices,dtype=opt.dtype)
    # Loop over ensemble:
    ensemble_predictions = []
    ensemble_residuals = []
    m_eff = -1
    for m in range(ensemble_size):

        if single_idx < 0 or (single_idx >=0 and m==single_idx):
          current_predictions = []
          for r in range(n_ranks):
            state_dict_path = f"{result_loc}_v{m}/models/generator_rank{r}.pt"
            current_state_dict = torch.load(state_dict_path,map_location=opt.devices)
            current_state_dict.pop("dataset_idx", None)
            opt.model.load_state_dict(current_state_dict)
            current_predictions.append(opt.predict(noise))
        
          current_predictions = torch.mean(torch.stack(current_predictions),0)
          current_residuals = true_predictions - current_predictions
          ensemble_predictions.append(current_predictions)
          ensemble_residuals.append(current_residuals)
          m_eff += 1
        
    ensemble_predictions = torch.stack(ensemble_predictions)    
    ensemble_mean = torch.mean(ensemble_predictions,0)
    ensemble_residuals = torch.stack(ensemble_residuals)
    ensemble_residual_mean = torch.mean(ensemble_residuals,0)
    ensemble_err = None
    ensemble_residual_err = None
    if m_eff > 0:
        ensemble_err = torch.std(ensemble_predictions,0)
        ensemble_residual_err = torch.std(ensemble_residuals,0)
    else:
        ensemble_err = torch.std(ensemble_predictions,1)
        ensemble_residual_err = torch.std(ensemble_residuals,1)

    return {
        'mean': ensemble_mean,
        'err': ensemble_err,
        'residual_mean': ensemble_residual_mean,
        'residual_err': ensemble_residual_err,
        'truth': true_predictions
    }

# Gather predictions from the MCMC model:
def gather_mc_prediction(result_loc,n_repetitions,n_ranks,batch_size,opt,env):
    # Get the true predictions:
    true_predictions = env.data_parser.true_pdfs
    # Create noise:
    noise = torch.normal(mean=0.0,std=1.0,size=(batch_size,opt.config.nz,1,1),device=opt.devices,dtype=opt.dtype)
    # Load the model from each rank:
    mc_predictions = None
    for r in range(n_ranks):
        state_dict_path = f"{result_loc}/models/generator_rank{r}.pt"
        current_state_dict = torch.load(state_dict_path,map_location=opt.devices)
        opt.model.load_state_dict(current_state_dict)

        # Tell the model that it should use the Dropout:
        for m in opt.model.modules():
          if isinstance(m, torch.nn.Dropout2d) or isinstance(m, ConcreteDropout):
             m.train()
          else:
             m.eval()

        current_predictions = opt.predict(noise,n_repetitions)

        if r == 0:
            mc_predictions = current_predictions / float(n_ranks)
        else:
            mc_predictions += current_predictions / float(n_ranks)

    mc_residuals = env.data_parser.true_pdfs - mc_predictions

    return {
        'mean': mc_predictions.mean(dim=0),
        'err': torch.sqrt(mc_predictions.var(dim=0)),
        'residual_mean': mc_residuals.mean(dim=0),
        'residual_err': torch.sqrt(mc_residuals.var(dim=0)),
        'truth': true_predictions
    }


#     mean_img = torch.sigmoid(mc_predictions.mean(dim=0))
#     sig_prime = mean_img * (1 - mean_img)
#    # var_img = (sig_prime ** 2) * mc_predictions.var(dim=0)
#     var_img = mc_predictions.var(dim=0)
#     return {
#         'mean': mean_img,
#         'err': torch.sqrt(var_img),
#         'residual_mean': mean_img,
#         'residual_err': torch.sqrt(var_img),
#         'truth': true_predictions
#     }
        

if __name__ == "__main__":
    run()
    
