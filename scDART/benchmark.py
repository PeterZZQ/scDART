import numpy as np
import pandas as pd 

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt

import torch

def kendalltau(pt_pred, pt_true):
    """\
    Description
        kendall tau correlationship
    
    Parameters
    ----------
    pt_pred
        inferred pseudo-time
    pt_true
        ground truth pseudo-time
    Returns
    -------
    tau
        returned score
    """
    from scipy.stats import kendalltau
    pt_true = pt_true.squeeze()
    pt_pred = pt_pred.squeeze()
    tau, p_val = kendalltau(pt_pred, pt_true)
    return tau

def get_k_neigh_ind(X,k):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)
    neigh_dist, neigh_ind = neigh.kneighbors(X) 
    return neigh_dist, neigh_ind


def alignscore(z1, z2, k = None):
    dsize = np.min([z1.shape[0],z2.shape[0]])
    if z1.shape[0] > dsize:
        idx = np.random.choice(np.arange(z1.shape[0]), size = dsize)
        z1 = z1[idx,:]
    elif z2.shape[0] > dsize:
        idx = np.random.choice(np.arange(z2.shape[0]), size = dsize)
        z2 = z2[idx,:]
    
    z = np.concatenate((z1, z2), axis = 0)
    if k is None:
        k = int(0.1 * (2 * dsize))
    
    _, neigh_ind = get_k_neigh_ind(z, k)
    z1_neigh = neigh_ind[:dsize,:]
    z2_neigh = neigh_ind[dsize:,:]
    # average number of neighbor belongs to the same modality
    x_bar = (np.sum((z1_neigh < dsize)) + np.sum((z2_neigh >= dsize)))/(2 * dsize)
    score = 1 - (x_bar-k/(2 * dsize))/(k - k/(2 * dsize))
    return score


def branching_acc(z_rna, z_atac, anno_rna, anno_atac, k = None):
    score = alignscore(z_rna, z_atac, k)
    branches = np.unique(anno_rna)
    print(branches)
    score_mtx = np.zeros((branches.shape[0], branches.shape[0]))

    for i, branch1 in enumerate(branches):
        bindx_rna = np.where(anno_rna == branch1)[0]
        b_z_rna = z_rna[bindx_rna, :]
        for j, branch2 in enumerate(branches):
            bindx_atac = np.where(anno_atac == branch2)[0]
            b_z_atac = z_atac[bindx_atac, :]
            score_mtx[i,j] = alignscore(b_z_rna, b_z_atac, k)
    
    return score, score_mtx

def neigh_overlap(z_rna, z_atac, k = 30):
    dsize = z_rna.shape[0]
    _, neigh_ind = get_k_neigh_ind(np.concatenate((z_rna, z_atac), axis = 0), k = k)
#     print(neigh_ind)
    z1_z2 = ((neigh_ind[:dsize,:] - dsize - np.arange(dsize)[:, None]) == 0)
#     print(z1_z2)
    z2_z1 = (neigh_ind[dsize:,:] - np.arange(dsize)[:, None] == 0)
#     print(z2_z1)
    return 0.5 * (np.sum(z1_z2) + np.sum(z2_z1))/dsize



# def align_fraction(data1, data2):
# 	row1, col1 = np.shape(data1)
# 	row2, col2 = np.shape(data2)
# 	fraction = 0
# 	for i in range(row1):
# 		count = 0
# 		diffMat = np.tile(data1[i], (row2,1)) - data2
# 		sqDiffMat = diffMat**2
# 		sqDistances = sqDiffMat.sum(axis=1)
# 		for j in range(row2):
# 			if sqDistances[j] < sqDistances[i]:
# 				count += 1
# 		fraction += count / row2

# 	return fraction / row1

# def transfer_accuracy(domain1, domain2, type1, type2):
# 	knn = KNeighborsClassifier()
# 	knn.fit(domain2, type2)
# 	type1_predict = knn.predict(domain1)
# 	np.savetxt("type1_predict.txt", type1_predict)
# 	count = 0
# 	for label1, label2 in zip(type1_predict, type1):
# 		if label1 == label2:
# 			count += 1
# 	return count / len(type1)

def gact_acc(gact, gact_true):
    diff = torch.sum(torch.logical_xor(gact, gact_true)).item()
    err = diff/(gact_true.shape[0] * gact_true.shape[1])
    return err