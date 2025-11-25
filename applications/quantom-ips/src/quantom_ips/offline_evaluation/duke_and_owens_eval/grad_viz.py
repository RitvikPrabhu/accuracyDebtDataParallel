import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size":20})

def show_gradients(data_dir,n_ranks,result_loc=None,model_performance_name="model_performance",avg_gradient_name="average_gradients",max_gradient_name="maximum_gradients",min_gradient_name="minimum_gradients",gradient_epochs_name="gradient_snapshot_epochs"):
    avg_grads = 0.0
    min_grads = 0.0
    max_grads = 0.0
    #++++++++++++++++++++++++
    for r in range(n_ranks):
        full_data_path = data_dir+"/"+model_performance_name+"_rank"+str(r)
        avg_grads += np.load(full_data_path+"/"+avg_gradient_name+"_rank"+str(r)+".npy")
        min_grads += np.load(full_data_path+"/"+min_gradient_name+"_rank"+str(r)+".npy")
        max_grads += np.load(full_data_path+"/"+max_gradient_name+"_rank"+str(r)+".npy")
        epochs = np.load(full_data_path+"/"+gradient_epochs_name+"_rank"+str(r)+".npy")
    #++++++++++++++++++++++++

    results = {
        'Minimum':min_grads/float(n_ranks),
        'Maximum':max_grads/float(n_ranks),
        'Average':avg_grads/float(n_ranks),
        'epochs':epochs
    }
    
    n_layers = results['Minimum'].shape[1]
    y_size = 6 + 2*n_layers

    fig,ax = plt.subplots(n_layers,1,figsize=(12,y_size),sharex=True)
    fig.suptitle("Generator Gradients")
    
    #++++++++++++++++++++++++
    for l in range(n_layers):
        for key, grads in results.items():
            if "epochs" not in key:
                ax[l].plot(epochs,grads[:,l],linewidth=3.0,label=key)
        
        ax[l].set_ylabel(f'Layer {l}')
        ax[l].grid(True)
        ax[l].legend(fontsize=15)
    #++++++++++++++++++++++++
    ax[n_layers-1].set_xlabel('Epochs')

    if result_loc is not None:
        fig.tight_layout()
        fig.savefig(result_loc+"/generator_gradients.png")
        plt.close(fig)

    return results
