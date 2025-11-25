import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import tqdm
import imageio

pdf_titles = {
    0: [r'$\rho_{0}^{gen}(x,y)$',r'$\rho_{0}^{real}(x,y)$',r'$\rho_{0}^{gen}(x,y)/\rho_{0}^{real}(x,y)$'],
    1: [r'$\rho_{1}^{gen}(x,y)$',r'$\rho_{1}^{real}(x,y)$',r'$\rho_{1}^{gen}(x,y)/\rho_{1}^{real}(x,y)$'],
    2: [r'$\rho_{2}^{gen}(x,y)$',r'$\rho_{2}^{real}(x,y)$',r'$\rho_{2}^{gen}(x,y)/\rho_{2}^{real}(x,y)$']
}

event_paths = {}
pdf_paths = {}

# Helper function to translate a sequence of .png to a movie:
def png_to_gif(image_paths,movie_path,movie_name,movie_duration):
        filenames = sorted(image_paths)
        images = []
        for filename in filenames:
          img = imageio.imread(filename)
          images.append(img)
        
        imageio.mimsave(os.path.join(movie_path,movie_name+'.gif'),images, duration=movie_duration)


# Define the pipeline which translates the predicted PDFs to events and store the results while doing so:
def run_viz_pipeline(torch_device,true_densities,predictions,env,n_bins,event_loc,pdf_loc,epoch_identifier,pdf_cmap,pdf_delta_r):
    
    predictions_torch = torch.as_tensor(predictions,device=torch_device)
    _, generated_events = env.step(predictions_torch)
    del predictions_torch
    if torch_device.lower() == "cuda":
        torch.cuda.empty_cache() 
    generated_events = generated_events.detach().cpu().numpy()
    # Sample real events from data:
    real_events = env.data_parser.get_samples(
        generated_events.shape, to_torch_tensor=False
    )
    n_data_sets = real_events.shape[1]
    n_densities = predictions.shape[1]

    # Plot and store events:
    #++++++++++++++++++++++
    for i in range(n_data_sets):
        fig_d, ax_d = plt.subplots(2,2,figsize=(12,12))
        fig_d.subplots_adjust(hspace=0.5,wspace=0.5)
        fig_d.suptitle(f"Real vs. Generated Events for Data {i} at {epoch_identifier}")

        ax_d[0,0].hist(real_events[0,i,:,0],n_bins,histtype='step',linewidth=3.0,color='black',label='Real')
        ax_d[0,0].hist(generated_events[0,i,:,0],n_bins,histtype='step',linewidth=3.0,color='red',label='Gen')
        ax_d[0,0].grid(True)
        ax_d[0,0].legend(fontsize=15)
        ax_d[0,0].set_xlabel('x')
        ax_d[0,0].set_ylabel('Entries') 

        ax_d[0,1].hist(real_events[0,i,:,1],n_bins,histtype='step',linewidth=3.0,color='black',label='Real')
        ax_d[0,1].hist(generated_events[0,i,:,1],n_bins,histtype='step',linewidth=3.0,color='red',label='Gen')
        ax_d[0,1].grid(True)
        ax_d[0,1].legend(fontsize=15)
        ax_d[0,1].set_xlabel('y')
        ax_d[0,1].set_ylabel('Entries')
        
        ax_d[1,0].set_title('Real')
        ax_d[1,0].hist2d(real_events[0,i,:,0],real_events[0,i,:,1],n_bins,norm=LogNorm()) 
        ax_d[1,0].grid(True)
        ax_d[1,0].set_xlabel('x')
        ax_d[1,0].set_ylabel('y')
        
        ax_d[1,1].set_title('Generated')
        ax_d[1,1].hist2d(generated_events[0,0,:,0],generated_events[0,0,:,1],n_bins,norm=LogNorm()) 
        ax_d[1,1].grid(True)
        ax_d[1,1].set_xlabel('x')
        ax_d[1,1].set_ylabel('y')
        
        if event_loc is not None:
            fig_d.savefig(f"{event_loc}/events_data{i}_{epoch_identifier}.png")

            if i in event_paths:
                event_paths[i].append(f"{event_loc}/events_data{i}_{epoch_identifier}.png")
            else:
                event_paths[i] = [f"{event_loc}/events_data{i}_{epoch_identifier}.png"]
        plt.close(fig_d)
    #++++++++++++++++++++++

    # Now Plot the PDFs:
    avg_densities = np.mean(
        predictions,
        axis=0
    )

    # Scale predictions w.r.t theory scalings:
    a_min = env.theory.a_min
    a_max = env.theory.a_max
    #++++++++++++++++++++++
    for j in range(n_densities):
       # Scale the predictions via the theory scaling --> To match the range of the
       # true PDF:
       current_gen_density = (a_max[j]-a_min[j]) * avg_densities[j] + a_min[j]
       # Compute the ratio between predicted and true:
       ratio = current_gen_density / true_densities[j]

       fig_p, ax_p = plt.subplots(1,3,figsize=(16,7),sharey=True,sharex=True)
       fig_p.suptitle(f"Densities {j} at {epoch_identifier}")
       fig_p.subplots_adjust(top=0.8)
       
       titles = pdf_titles[j]
       
       ax_p[0].set_title(titles[0])
       ax_p[0].pcolormesh(current_gen_density,cmap=pdf_cmap)
       ax_p[0].set_xlabel('x')
       ax_p[0].set_ylabel('y')
       ax_p[0].grid(True)
       
       ax_p[1].set_title(titles[1])
       ax_p[1].pcolormesh(true_densities[j],cmap=pdf_cmap)
       ax_p[1].set_xlabel('x')
       ax_p[1].grid(True)

       ax_p[2].set_title(titles[2])
       v_min = 1.0 - pdf_delta_r
       v_max = 1.0 + pdf_delta_r
       img = ax_p[2].pcolormesh(ratio,vmin=v_min,vmax=v_max,cmap=pdf_cmap)
       ax_p[2].set_xlabel('x')
       ax_p[2].grid(True)

       fig_p.colorbar(img, ax=ax_p.ravel().tolist(), orientation='vertical')
       if pdf_loc is not None:
            fig_p.savefig(f"{pdf_loc}/density{j}_{epoch_identifier}.png")

            if j in pdf_paths:
                pdf_paths[j].append(f"{pdf_loc}/density{j}_{epoch_identifier}.png")
            else:
                pdf_paths[j] = [f"{pdf_loc}/density{j}_{epoch_identifier}.png"]
       plt.close(fig_p)
    #++++++++++++++++++++++

# Analyze a single epoch:
def analyze_single_epoch(noise,epoch_idx,n_epochs,opt,env,true_densities,ensemble_size,n_ranks,data_dir,n_bins,pdf_cmap,pdf_delta_r,event_loc,pdf_loc,model_snapshot_name,generator_name,ensemble_postfix):
    # Create epoch string as unique name identifier:
    epoch_str = str(epoch_idx) + "epochs"
    epoch_str = epoch_str.zfill(6 + len(str(n_epochs)))

    ensemble_predictions = []
    # Loop over ensemble:
    #++++++++++++++++++++
    for m in range(ensemble_size):
        current_predictions = None
        # Now loop over all ranks:
        #++++++++++++++++++++
        for r in range(n_ranks):
           # Get path of the current model that was stored at rank r and epoch_idx:
            model_loc = f"{data_dir}{ensemble_postfix}{m}/{model_snapshot_name}_rank{r}/{generator_name}_rank{r}_{epoch_str}.pt"
            # Load the model states:
            current_state_dict = torch.load(model_loc,map_location=opt.devices)
            # Check if we have an EMA-model that uses slightly different keys
            new_dict = {}
            for key in current_state_dict:
              if 'module' in key:
                  new_key = key.replace('module.','')
                  new_dict[new_key] = current_state_dict[key]
              else: 
                  new_dict[key] = current_state_dict[key]

            # Overwrite weights in optimizer:
            opt.model.load_state_dict(new_dict)
            #Accumulate predictions over ranks:

            if r == 0:
               current_predictions = opt.predict(noise).detach().cpu().numpy() / float(n_ranks)
            else:
               current_predictions += opt.predict(noise).detach().cpu().numpy() / float(n_ranks)

            if str(opt.devices) == "cuda":
               torch.cuda.empty_cache() 
        #++++++++++++++++++++
        ensemble_predictions.append(current_predictions)
    #++++++++++++++++++++

    predictions = np.stack(ensemble_predictions)
    predictions_mean = np.mean(predictions,0)
    predictions_err = None
    if ensemble_size > 1:
        predictions_err = np.std(predictions,0)

    # Now that we have the generator predictions, averaged over ranks, 
    # we may go ahead and feed then into the environment and create nice plots:
    run_viz_pipeline(
        opt.devices,
        true_densities,
        predictions_mean,
        env,
        n_bins,
        event_loc,
        pdf_loc,epoch_str,
        pdf_cmap,
        pdf_delta_r
    )
    
    # Delete what we do not need:
    del predictions
    del ensemble_predictions
    if str(opt.devices) == "cuda":
        torch.cuda.empty_cache() 
    return predictions_mean, predictions_err, epoch_str


# Analyze the trained generators:
def analyze_trained_generators(data_dir,ensemble_size,n_ranks,batch_size,true_densities,opt,env,n_bins,pdf_cmap,pdf_delta_r,movie_duration,result_loc=None,model_performance_name="model_performance",model_snapshot_name="model_snapshots",generator_name="optimizer",snapshot_epochs_name="model_snapshot_epochs",ensemble_postfix="_v"):
    # First get the epochs where the model snapshots have been taken.
    # The interval was the same for all ranks and every ensemble, so we can just load the stored epochs for the first rank and ensemble:
    epochs = np.load(data_dir+str(ensemble_postfix)+"0/"+model_performance_name+"_rank0/"+snapshot_epochs_name+"_rank0.npy")
    n_epochs = epochs[-1]

    # Create common noise that all generators will see:
    noise = torch.normal(mean=0.0,std=1.0,size=(batch_size,opt.input_shape[1]),device=opt.devices)

    # Create directories to store our results:
    event_loc = None
    pdf_loc = None
    npy_loc = None
    if result_loc is not None:
        event_loc = result_loc+"/events"
        pdf_loc = result_loc+"/densities"
        npy_loc = result_loc+"/npy_data"
        os.makedirs(event_loc,exist_ok=True)
        os.makedirs(pdf_loc,exist_ok=True)
        os.makedirs(npy_loc,exist_ok=True)

    # Now that we have the epochs, we can simply loop over them and run the single epoch analysis:
    pbar = tqdm.tqdm(range(epochs.shape[0]))
    #+++++++++++++++++++++++++++
    for ep in pbar:
        pred_mean, pred_err, epoch_str = analyze_single_epoch(
            noise,
            epochs[ep],
            n_epochs,
            opt,
            env,
            true_densities,
            ensemble_size,
            n_ranks,
            data_dir,
            n_bins,
            pdf_cmap,
            pdf_delta_r,
            event_loc,
            pdf_loc,
            model_snapshot_name,
            generator_name,
            ensemble_postfix
        )

        # Write predictions / errors to file for every epoch:
        np.save(f"{npy_loc}/avg_densities_{epoch_str}.npy",pred_mean)
        if pred_err is not None:
            np.save(f"{npy_loc}/err_densities_{epoch_str}.npy",pred_err)
    #+++++++++++++++++++++++++++

    # Write out the true densities, so that we can use them for comparison:
    np.save(f"{npy_loc}/true_densities.npy",true_densities)

    # Translate the pngs that we generated in gifs, so that we have something interesting to present:
    if bool(event_paths):
        for key, images in event_paths.items():
            movie_name = f"evolution_events{key}"
            png_to_gif(images,event_loc,movie_name,movie_duration)

    if bool(pdf_paths):
        for key, images in pdf_paths.items():
            movie_name = f"evolution_densities{key}"
            png_to_gif(images,pdf_loc,movie_name,movie_duration)
