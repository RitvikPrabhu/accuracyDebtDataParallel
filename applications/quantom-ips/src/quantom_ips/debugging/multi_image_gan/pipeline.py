import numpy as np
import torch

def pipeline(predictions,alphas,is_torch):
    a_0 = alphas[0][0]*predictions[0] + alphas[0][1]*predictions[1]
    a_1 = alphas[1][0]*predictions[0] + alphas[1][1]*predictions[1]
    all_data = [a_0,a_1]
    
    if is_torch:
       return torch.stack(all_data)
    
    return np.stack(all_data)
