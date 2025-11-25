import torch 
import numpy as np
from quantom_ips.utils.log_sad_score import compute_logSAD

# Collect the predictions of each model within the ensemble
def collect_individual_predictions(model_loc,ensemble_size,n_ranks,t_start,t_step,batch_size,opt,i_max,model_performance_name="model_performance",model_snapshot_name="model_snapshots",generator_name="optimizer",snapshot_epochs_name="model_snapshot_epochs",snapshot_times_name="model_snapshot_times"):
    t_i = 0 
    min_idx = 0
    collection_not_finished = True
    collected_predictions = [[None]*n_ranks]*ensemble_size
    all_predictions = []
    collected_times = []
    
    while collection_not_finished and t_i < i_max:
      current_t = t_start + t_i * t_step
      
      # Loop over ensemble:
      #+++++++++++++++++
      for m in range(ensemble_size):
        model_path = model_loc+"_v"+str(m)

        # Loop over ranks:
        #+++++++++++++++++ 
        for r in range(n_ranks):
            epochs = np.load(model_path+"/"+model_performance_name+"_rank"+str(r)+"/"+snapshot_epochs_name+"_rank"+str(r)+".npy")
            times = np.load(model_path+"/"+model_performance_name+"_rank"+str(r)+"/"+snapshot_times_name+"_rank"+str(r)+".npy")
            
            n_epochs = epochs[-1]
            acc_times = times < current_t
            if acc_times[min_idx]:
               epoch_str = str(epochs[min_idx]) + "epochs"
               epoch_str = epoch_str.zfill(6 + len(str(n_epochs)))
               current_state_dict = torch.load(model_path+"/"+model_snapshot_name+"_rank"+str(r)+"/"+generator_name+"_rank"+str(r)+"_"+epoch_str+".pt",map_location=opt.devices)
               opt.model.load_state_dict(current_state_dict)
               collected_predictions[m][r] = torch.mean(opt.forward(batch_size),(0,1)).detach().cpu().numpy()
        #+++++++++++++++++
      #+++++++++++++++++

      # Check if all epochs have been collected:
      current_collection_finished = True
      #+++++++++++++++++
      for single in collected_predictions:
         for el in single:
            if el is None:
              current_collection_finished = False
      #+++++++++++++++++
      
      # Once all None entries are gone, we may collect all predictions made for the current epoch
      if current_collection_finished:
         all_predictions.append(collected_predictions)
         collected_times.append(current_t)
         collected_predictions = [[None]*n_ranks]*ensemble_size
         min_idx += 1

      t_i += 1
      if min_idx >= epochs.shape[0]:
        collection_not_finished = False

    return np.array(all_predictions), np.array(collected_times)


# Turn the individual predictions into and ensemble prediction with mean and std:
def get_ensemble_predictions(individual_predictions):
    # First, compute the predictions over all ranks:
    avg_preds_over_ranks = np.mean(individual_predictions,axis=2)
    # Second, compute mean and std. of the ensemble predictions:
    ensemble_mean = np.mean(avg_preds_over_ranks,1)
    ensemble_std = np.std(avg_preds_over_ranks,1)

    return np.squeeze(ensemble_mean), np.squeeze(ensemble_std)

# Once we have the ensemble mean and std. we may proceed and compute a few metrics that hopefuly give us insights into the models performance:
def get_metrics(mean,std,true_img,image_axes=(1,2)):
   # We are dealing with images here, so we may just compute the mean of the image dimensions:
   avg = np.mean(mean,axis=image_axes)
   err = np.mean(std,axis=image_axes)

   # Now we are computing a chi^2 even though it is, strictly speaking, not a true chi^2
   chi2 = np.square((true_img-mean) / std)
   chi2_per_ndf = np.mean(chi2,axis=image_axes)

   # Lastly, compute the logSAD score:
   log_sad = compute_logSAD(true_img,mean,axis=(1,2))
   log_sad_low = compute_logSAD(true_img,mean-std,axis=(1,2))
   log_sad_up = compute_logSAD(true_img,mean+std,axis=(1,2))

   return {
      'true_avg': np.mean(true_img),
      'ensemble_avg':avg,
      'ensemble_err':err,
      'log_sad': log_sad,
      'log_sad_low':log_sad_low,
      'log_sad_up':log_sad_up,
      'chi2_per_ndf':chi2_per_ndf
   }
