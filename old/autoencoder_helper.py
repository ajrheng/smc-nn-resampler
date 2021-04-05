import numpy as np
import torch

def gaussian_1d(x, mean, std, tensor=True):
    if tensor:
        return 1/np.sqrt(2*np.pi*std**2) * torch.exp(-(x-mean)**2/(2*std**2))
    else:
        return 1/np.sqrt(2*np.pi*std**2) * np.exp(-(x-mean)**2/(2*std**2))

def likelihood_loss(prediction, i, stats, batch=True):
    
    if batch:
        data_stats = stats[i*batch_size:(i+1)*batch_size, :]
    else:
        data_stats = stats
            
    for j in range(data_stats.shape[0]):
        mean = data_stats[j,0]
        stddev = data_stats[j,1]
        pdf_vals = gaussian_1d(stddev*prediction[j,:]+mean, mean, stddev)
        _ll = -torch.mean(torch.log(pdf_vals)).reshape(1) # so it is not a zero-dimensional tensor
        if j == 0:
            ll = torch.clone(_ll)
        else:
            ll = torch.cat((ll,_ll))

    ll_val = torch.mean(ll)
    return ll_val
