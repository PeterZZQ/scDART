import numpy as np 
import pandas as pd 
import torch 


def _pairwise_distances(x, y = None):
    x_norm = (x**2).sum(1).view(-1, 1)
    # calculate the pairwise distance between two datasets
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    if y is None:
        dist = dist - torch.diag(dist.diag)

    return torch.clamp(dist, 0.0, np.inf)

def match_alignment(z_rna, z_atac, k = 10):
    # note that the distance is squared version
    dist = _pairwise_distances(z_atac, z_rna).numpy()
    knn_index = np.argpartition(dist, kth = k - 1, axis = 1)[:,(k-1)]
    kth_dist = np.take_along_axis(dist, knn_index[:,None], axis = 1)
    
    K = dist/kth_dist 
    K = (dist <= kth_dist) * np.exp(-K) 
    K = K/np.sum(K, axis = 1)[:,None]

    z_atac = torch.FloatTensor(K).mm(z_rna)
    return z_rna, z_atac
