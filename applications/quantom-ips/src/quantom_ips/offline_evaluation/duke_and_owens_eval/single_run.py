import torch
import numpy as np
from quantom_ips.utils.log_sad_score import compute_logSAD, get_logSAD_from_events
from quantom_ips.utils.pixelated_densities import DukeAndOwensDensities
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

plt.rcParams.update({'font.size':20})


# Get metric from name:
def get_metric_from_name(data_path,metric_name,rank):
    if metric_name is not None and metric_name != "":
        try:
           full_metric = data_path+"/"+metric_name+"_rank"+str(rank) + ".npy"
           return np.load(full_metric)
        except:
            print(f"Metric: {metric_name} not available.")
            return 0.0
    return 0.0


# Get average performance vs. average time:
def get_avg_performance_vs_avg_time(data_dir,n_ranks,model_performance_name="model_performance",time_name="train_step_time",loss_names={'real_loss':"real_loss",'gen_loss':"gen_loss",'fake_loss':"fake_loss"},score_name="log_sad_score",epochs_name="epochs"):
    avg_real_loss = 0.0
    avg_gen_loss = 0.0
    avg_fake_loss = 0.0 
    avg_training_time = 0.0
    avg_score = 0.0
    epochs = 0
    #+++++++++++++++++++++++++++++++++
    for r in range(n_ranks):
        full_data_path = data_dir+"/"+model_performance_name+"_rank"+str(r)
        avg_training_time += get_metric_from_name(full_data_path,time_name,r)/n_ranks
        avg_score += get_metric_from_name(full_data_path,score_name,r)/n_ranks
        avg_real_loss += get_metric_from_name(full_data_path,loss_names['real_loss'],r)/n_ranks
        avg_fake_loss += get_metric_from_name(full_data_path,loss_names['fake_loss'],r)/n_ranks
        avg_gen_loss += get_metric_from_name(full_data_path,loss_names['gen_loss'],r)/n_ranks
        epochs = get_metric_from_name(full_data_path,epochs_name,r)
    #+++++++++++++++++++++++++++++++++

    results = {
      'real_loss': avg_real_loss,
      'fake_loss':avg_fake_loss,
      'gen_loss':avg_gen_loss,
      'epochs': epochs,
      'times':avg_training_time
    }
    results[score_name] = avg_score
    return results


# Analyze the trained generators:
def analyze_trained_generators(data_dir,n_ranks,batch_size,true_densities,opt,env,n_bins,result_loc=None,model_performance_name="model_performance",model_snapshot_name="model_snapshots",generator_name="optimizer",snapshot_epochs_name="model_snapshot_epochs"):
    u_scores = 0.0
    d_scores = 0.0
    collected_densities = []
    #++++++++++++++++++++++++++++
    for r in range(n_ranks):
        # Get the number of epochs first:
        epochs = np.load(data_dir+"/"+model_performance_name+"_rank"+str(r)+"/"+snapshot_epochs_name+"_rank"+str(r)+".npy")
        n_epochs = epochs[-1]
        
        current_scores_u = []
        current_scores_d = []
        current_densities = []
        # Now go through each epoch collect the generator snapshots,
        # make a prediction and then collect the predictions as a function of time:
        #++++++++++++++++++++++++++++
        for epoch in epochs:
            epoch_str = str(epoch) + "epochs"
            epoch_str = epoch_str.zfill(6 + len(str(n_epochs)))
            # Get the model loc:
            model_loc = data_dir+"/"+model_snapshot_name+"_rank"+str(r)+"/"+generator_name+"_rank"+str(r)+"_"+epoch_str+".pt"
            # Load the model weights:
            current_state_dict = torch.load(model_loc,map_location=opt.devices)
            new_dict = {}
            for key in current_state_dict:
                if 'module' in key:
                    new_key = key.replace('module.','')
                    new_dict[new_key] = current_state_dict[key]
                else: 
                 new_dict[key] = current_state_dict[key]


            # Overwrite weights in optimizer:
            opt.model.load_state_dict(new_dict)
            # Get the predictions:
            predictions = opt.forward(batch_size)
            np_predictions = predictions.detach().cpu().numpy()
            predictions_mean = np.mean(np_predictions,0)
            # Compute log[SAD] score:
            current_log_sad_u = compute_logSAD(true_densities[0],predictions_mean[0])
            current_log_sad_d = compute_logSAD(true_densities[1],predictions_mean[1])
            current_scores_u.append(current_log_sad_u)
            current_scores_d.append(current_log_sad_d)
            current_densities.append(predictions_mean)
        #++++++++++++++++++++++++++++
        u_scores += np.array(current_scores_u)/n_ranks
        d_scores += np.array(current_scores_d)/n_ranks
        collected_densities.append(np.stack(current_densities))
    #++++++++++++++++++++++++++++

    all_densities = np.stack(collected_densities)

    print(f"all-densities-shape: {all_densities.shape}")
    print(f"all-real-data: {env.data_parser.data.shape}")

    # Compute the mean accross all ranks:
    avg_densities = np.mean(all_densities,0)

    events_dir = None
    if result_loc is not None:
        events_dir = result_loc+"/events"
        os.makedirs(events_dir,exist_ok=True)

    # Now generate events for each epoch:
    proton_scores = []
    neutron_scores = []
    #++++++++++++++++++++++++
    for ep in range(avg_densities.shape[0]):
        epoch_str = str(epochs[ep]) + "epochs"
        epoch_str = epoch_str.zfill(6 + len(str(n_epochs)))

        _, generated_events = env.step(torch.as_tensor(avg_densities[ep][None,:],device=env.devices))
        generated_events = generated_events.detach().cpu().numpy()
        # Sample real events from data:
        real_events = env.data_parser.get_samples(
            generated_events.shape, to_torch_tensor=False
        )

        # Compute the log sad score for each data type:
        log_sad_proton = get_logSAD_from_events(real_events[0,0,:,:],generated_events[0,0,:,:],n_bins)
        log_sad_neutron = get_logSAD_from_events(real_events[0,1,:,:],generated_events[0,1,:,:],n_bins)
        proton_scores.append(log_sad_proton)
        neutron_scores.append(log_sad_neutron)

        log_real_events = np.log(real_events)
        log_generated_events = np.log(generated_events)

        # Proton data:
        fig_p, ax_p = plt.subplots(2,2,figsize=(12,12))
        fig_p.subplots_adjust(hspace=0.5,wspace=0.5)
        fig_p.suptitle('Real vs. Generated Events for Proton Target at '+epoch_str)

        ax_p[0,0].hist(log_real_events[0,0,:,0],n_bins,histtype='step',linewidth=3.0,color='black',label='Real')
        ax_p[0,0].hist(log_generated_events[0,0,:,0],n_bins,histtype='step',linewidth=3.0,color='red',label='Gen')
        ax_p[0,0].grid(True)
        ax_p[0,0].legend(fontsize=15)
        ax_p[0,0].set_xlabel(r'$log(x)$')
        ax_p[0,0].set_ylabel('Entries') 

        ax_p[0,1].hist(log_real_events[0,0,:,1],n_bins,histtype='step',linewidth=3.0,color='black',label='Real')
        ax_p[0,1].hist(log_generated_events[0,0,:,1],n_bins,histtype='step',linewidth=3.0,color='red',label='Gen')
        ax_p[0,1].grid(True)
        ax_p[0,1].legend(fontsize=15)
        ax_p[0,1].set_xlabel(r'$log(Q^{2})$')
        ax_p[0,1].set_ylabel('Entries')
        
        ax_p[1,0].set_title('Real')
        ax_p[1,0].hist2d(log_real_events[0,0,:,0],log_real_events[0,0,:,1],n_bins,norm=LogNorm()) 
        ax_p[1,0].grid(True)
        ax_p[1,0].set_xlabel(r'$log(x)$')
        ax_p[1,0].set_ylabel(r'$log(Q^{2})$')
        
        ax_p[1,1].set_title('Generated')
        ax_p[1,1].hist2d(log_generated_events[0,0,:,0],log_generated_events[0,0,:,1],n_bins,norm=LogNorm()) 
        ax_p[1,1].grid(True)
        ax_p[1,1].set_xlabel(r'$log(x)$')
        ax_p[1,1].set_ylabel(r'$log(Q^{2})$')

        if result_loc is not None:
            fig_p.savefig(events_dir+"/events_proton_"+epoch_str+".png")
        plt.close(fig_p)

        # Neutron data:
        fig_n, ax_n = plt.subplots(2,2,figsize=(12,12))
        fig_n.subplots_adjust(hspace=0.5,wspace=0.5)
        fig_n.suptitle('Real vs. Generated Events for Neutron Target at '+epoch_str)

        ax_n[0,0].hist(log_real_events[0,1,:,0],n_bins,histtype='step',linewidth=3.0,color='black',label='Real')
        ax_n[0,0].hist(log_generated_events[0,1,:,0],n_bins,histtype='step',linewidth=3.0,color='red',label='Gen')
        ax_n[0,0].grid(True)
        ax_n[0,0].legend(fontsize=15)
        ax_n[0,0].set_xlabel(r'$log(x)$')
        ax_n[0,0].set_ylabel('Entries') 

        ax_n[0,1].hist(log_real_events[0,1,:,1],n_bins,histtype='step',linewidth=3.0,color='black',label='Real')
        ax_n[0,1].hist(log_generated_events[0,1,:,1],n_bins,histtype='step',linewidth=3.0,color='red',label='Gen')
        ax_n[0,1].grid(True)
        ax_n[0,1].legend(fontsize=15)
        ax_n[0,1].set_xlabel(r'$log(Q^{2})$')
        ax_n[0,1].set_ylabel('Entries')
        
        ax_n[1,0].set_title('Real')
        ax_n[1,0].hist2d(log_real_events[0,1,:,0],log_real_events[0,1,:,1],n_bins,norm=LogNorm()) 
        ax_n[1,0].grid(True)
        ax_n[1,0].set_xlabel(r'$log(x)$')
        ax_n[1,0].set_ylabel(r'$log(Q^{2})$')
        
        ax_n[1,1].set_title('Generated')
        ax_n[1,1].hist2d(log_generated_events[0,1,:,0],log_generated_events[0,1,:,1],n_bins,norm=LogNorm()) 
        ax_n[1,1].grid(True)
        ax_n[1,1].set_xlabel(r'$log(x)$')
        ax_n[1,1].set_ylabel(r'$log(Q^{2})$')

        if result_loc is not None:
            fig_n.savefig(events_dir+"/events_neutron_"+epoch_str+".png")
        plt.close(fig_n)

        # Plot the densities:
        density_dir = None
        if result_loc is not None:
            density_dir = result_loc+"/densities"
            os.makedirs(density_dir,exist_ok=True)
        
        # u:
        fig_u, ax_u = plt.subplots(1,2,figsize=(16,8),sharey=True,sharex=True)
        fig_u.suptitle('u-Quark Densities at '+epoch_str)

        ax_u[0].set_title('Generated')
        ax_u[0].pcolormesh(avg_densities[ep][0])
        ax_u[0].set_xlabel('x')
        ax_u[0].set_ylabel(r'$Q^{2}$')
        ax_u[0].grid(True)

        ax_u[1].set_title('Real')
        ax_u[1].pcolormesh(true_densities[0])
        ax_u[1].set_xlabel('x')
        ax_u[1].set_ylabel(r'$Q^{2}$')
        ax_u[1].grid(True)

        if density_dir is not None:
            fig_u.savefig(density_dir+"/u_densities_"+epoch_str+".png")
        plt.close(fig_u)
        
        # d:
        fig_d, ax_d = plt.subplots(1,2,figsize=(16,8),sharey=True,sharex=True)
        fig_d.suptitle('d-Quark Densities at '+epoch_str)

        ax_d[0].set_title('Generated')
        ax_d[0].pcolormesh(avg_densities[ep][1])
        ax_d[0].set_xlabel('x')
        ax_d[0].set_ylabel(r'$Q^{2}$')
        ax_d[0].grid(True)

        ax_d[1].set_title('Real')
        ax_d[1].pcolormesh(true_densities[1])
        ax_d[1].set_xlabel('x')
        ax_d[1].set_ylabel(r'$Q^{2}$')
        ax_d[1].grid(True)

        if density_dir is not None:
            fig_d.savefig(density_dir+"/d_densities_"+epoch_str+".png")
        plt.close(fig_d)
    #++++++++++++++++++++++++
    

    return {
        'epochs':epochs,
        'u_scores':u_scores,
        'd_scores':d_scores,
        'proton_scores':np.stack(proton_scores,0),
        'neutron_scores':np.stack(neutron_scores,0)
    }
