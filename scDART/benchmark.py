import numpy as np
import pandas as pd 
import os
import subprocess

import scipy.special
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import silhouette_samples, silhouette_score
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


def F1_branches(branches, branches_gt):
    recovery = 0
    #  Recovery is the average maximal Jaccard for every cluster in the first set of clusters (the reference trajectory)
    for branch_gt in np.sort(np.unique(branches_gt)):
        # find cells in ground truth branch
        cells_branch_gt = np.where(branches_gt == branch_gt)[0]
        cells_branch_gt = set([x for x in cells_branch_gt])

        max_jaccard = 0

        for branch in np.sort(np.unique(branches)):
            # find cells in the branch
            cells_branch = np.where(branches == branch)[0]
            cells_branch = set([x for x in cells_branch])
            jaccard = len(cells_branch.intersection(cells_branch_gt))/len(cells_branch.union(cells_branch_gt))
            max_jaccard = max([jaccard, max_jaccard])

        # calculate the maximum jaccard score
        # print("original trajectory: {:d}, Jaccard: {:.4f}".format(branch_gt, max_jaccard))
        recovery += max_jaccard

    recovery = recovery / np.sort(np.unique(branches_gt)).shape[0] 

    relevence = 0
    #  Relevence is the average maximal Jaccard for every cluster in the second set of clusters (the predict trajectory)
    for branch in np.sort(np.unique(branches)):
        # find cells in the branch
        cells_branch = np.where(branches == branch)[0]
        cells_branch = set([x for x in cells_branch])

        max_jaccard = 0

        for branch_gt in np.sort(np.unique(branches_gt)):
            # find cells in ground truth branch
            cells_branch_gt = np.where(branches_gt == branch_gt)[0]
            cells_branch_gt = set([x for x in cells_branch_gt])
            jaccard = len(cells_branch.intersection(cells_branch_gt))/len(cells_branch.union(cells_branch_gt))
            max_jaccard = max([jaccard, max_jaccard])

        # calculate the maximum jaccard score
        # print("inferred trajectory: {:d}, Jaccard: {:.4f}".format(branch, max_jaccard))
        relevence += max_jaccard

    relevence = relevence / np.sort(np.unique(branches)).shape[0] 

    F1 = 2/(1/recovery + 1/relevence)
    return F1




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



########################################################################################
#
# ARI score from scIB(https://github.com/theislab/scib/tree/main/scib)
#
########################################################################################

def ari(group1, group2, implementation=None):
    """ Adjusted Rand Index
    The function is symmetric, so group1 and group2 can be switched
    For single cell integration evaluation the scenario is:
        predicted cluster assignments vs. ground-truth (e.g. cell type) assignments
    :param adata: anndata object
    :param group1: string of column in adata.obs containing labels
    :param group2: string of column in adata.obs containing labels
    :params implementation: of set to 'sklearn', uses sklearns implementation,
        otherwise native implementation is taken
    """

    if len(group1) != len(group2):
        raise ValueError(
            f'different lengths in group1 ({len(group1)}) and group2 ({len(group2)})'
        )

    if implementation == 'sklearn':
        return adjusted_rand_score(group1, group2)

    def binom_sum(x, k=2):
        return scipy.special.binom(x, k).sum()

    n = len(group1)
    contingency = pd.crosstab(group1, group2)

    ai_sum = binom_sum(contingency.sum(axis=0))
    bi_sum = binom_sum(contingency.sum(axis=1))

    index = binom_sum(np.ravel(contingency))
    expected_index = ai_sum * bi_sum / binom_sum(n, 2)
    max_index = 0.5 * (ai_sum + bi_sum)

    return (index - expected_index) / (max_index - expected_index)




########################################################################################
#
# NMI score from scIB(https://github.com/theislab/scib/tree/main/scib)
#
########################################################################################
def nmi(group1, group2, method="arithmetic", nmi_dir=None):
    """
    Wrapper for normalized mutual information NMI between two different cluster assignments
    :param adata: Anndata object
    :param group1: column name of `adata.obs`
    :param group2: column name of `adata.obs`
    :param method: NMI implementation
        'max': scikit method with `average_method='max'`
        'min': scikit method with `average_method='min'`
        'geometric': scikit method with `average_method='geometric'`
        'arithmetic': scikit method with `average_method='arithmetic'`
        'Lancichinetti': implementation by A. Lancichinetti 2009 et al. https://sites.google.com/site/andrealancichinetti/mutual
        'ONMI': implementation by Aaron F. McDaid et al. https://github.com/aaronmcdaid/Overlapping-NMI
    :param nmi_dir: directory of compiled C code if 'Lancichinetti' or 'ONMI' are specified as `method`.
        These packages need to be compiled as specified in the corresponding READMEs.
    :return:
        Normalized mutual information NMI value
    """
    
    if len(group1) != len(group2):
        raise ValueError(
            f'different lengths in group1 ({len(group1)}) and group2 ({len(group2)})'
        )

    # choose method
    if method in ['max', 'min', 'geometric', 'arithmetic']:
        nmi_value = normalized_mutual_info_score(group1, group2, average_method=method)
    elif method == "Lancichinetti":
        nmi_value = nmi_Lanc(group1, group2, nmi_dir=nmi_dir)
    elif method == "ONMI":
        nmi_value = onmi(group1, group2, nmi_dir=nmi_dir)
    else:
        raise ValueError(f"Method {method} not valid")

    return nmi_value


def onmi(group1, group2, nmi_dir=None, verbose=True):
    """
    Based on implementation https://github.com/aaronmcdaid/Overlapping-NMI
    publication: Aaron F. McDaid, Derek Greene, Neil Hurley 2011
    params:
        nmi_dir: directory of compiled C code
    """

    if nmi_dir is None:
        raise FileNotFoundError(
            "Please provide the directory of the compiled C code from "
            "https://sites.google.com/site/andrealancichinetti/mutual3.tar.gz"
        )

    group1_file = write_tmp_labels(group1, to_int=False)
    group2_file = write_tmp_labels(group2, to_int=False)

    nmi_call = subprocess.Popen(
        [nmi_dir + "onmi", group1_file, group2_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)

    stdout, stderr = nmi_call.communicate()
    if stderr:
        print(stderr)

    nmi_out = stdout.decode()
    if verbose:
        print(nmi_out)

    nmi_split = [x.strip().split('\t') for x in nmi_out.split('\n')]
    nmi_max = float(nmi_split[0][1])

    # remove temporary files
    os.remove(group1_file)
    os.remove(group2_file)

    return nmi_max


def nmi_Lanc(group1, group2, nmi_dir="external/mutual3/", verbose=True):
    """
    paper by A. Lancichinetti 2009
    https://sites.google.com/site/andrealancichinetti/mutual
    recommended by Malte
    """

    if nmi_dir is None:
        raise FileNotFoundError(
            "Please provide the directory of the compiled C code from https://sites.google.com/site/andrealancichinetti/mutual3.tar.gz")

    group1_file = write_tmp_labels(group1, to_int=False)
    group2_file = write_tmp_labels(group2, to_int=False)

    nmi_call = subprocess.Popen(
        [nmi_dir + "mutual", group1_file, group2_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)

    stdout, stderr = nmi_call.communicate()
    if stderr:
        print(stderr)
    nmi_out = stdout.decode().strip()

    return float(nmi_out.split('\t')[1])


def write_tmp_labels(group_assignments, to_int=False, delim='\n'):
    """
    write the values of a specific obs column into a temporary file in text format
    needed for external C NMI implementations (onmi and nmi_Lanc functions), because they require files as input
    params:
        to_int: rename the unique column entries by integers in range(1,len(group_assignments)+1)
    """
    import tempfile

    if to_int:
        label_map = {}
        i = 1
        for label in set(group_assignments):
            label_map[label] = i
            i += 1
        labels = delim.join([str(label_map[name]) for name in group_assignments])
    else:
        labels = delim.join([str(name) for name in group_assignments])

    clusters = {label: [] for label in set(group_assignments)}
    for i, label in enumerate(group_assignments):
        clusters[label].append(str(i))

    output = '\n'.join([' '.join(c) for c in clusters.values()])
    output = str.encode(output)

    # write to file
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(output)
        filename = f.name

    return filename




########################################################################################
#
# silhouette score from scIB(https://github.com/theislab/scib/tree/main/scib)
#
########################################################################################

def silhouette(
        X,
        group_gt,
        metric='euclidean',
        scale=True
):
    """
    Wrapper for sklearn silhouette function values range from [-1, 1] with
        1 being an ideal fit
        0 indicating overlapping clusters and
        -1 indicating misclassified cells
    By default, the score is scaled between 0 and 1. This is controlled `scale=True`
    :param group_gt: cell labels
    :param X: embedding e.g. PCA
    :param scale: default True, scale between 0 (worst) and 1 (best)
    """
    asw = silhouette_score(
        X=X,
        labels=group_gt,
        metric=metric
    )
    if scale:
        asw = (asw + 1) / 2
    return asw


def silhouette_batch(
        X,
        batch_gt,
        group_gt,
        metric='euclidean',
        return_all=False,
        scale=True,
        verbose=True
):
    """
    Absolute silhouette score of batch labels subsetted for each group.
    :param batch_key: batches to be compared against
    :param group_key: group labels to be subsetted by e.g. cell type
    :param embed: name of column in adata.obsm
    :param metric: see sklearn silhouette score
    :param scale: if True, scale between 0 and 1
    :param return_all: if True, return all silhouette scores and label means
        default False: return average width silhouette (ASW)
    :param verbose:
    :return:
        average width silhouette ASW
        mean silhouette per group in pd.DataFrame
        Absolute silhouette scores per group label
    """

    sil_all = pd.DataFrame(columns=['group', 'silhouette_score'])

    for group in np.sort(np.unique(group_gt)):
        X_group = X[group_gt == group, :]
        batch_group = batch_gt[group_gt == group]
        n_batches = np.unique(batch_group).shape[0]

        if (n_batches == 1) or (n_batches == X_group.shape[0]):
            continue

        sil_per_group = silhouette_samples(
            X_group,
            batch_group,
            metric=metric
        )

        # take only absolute value
        sil_per_group = [abs(i) for i in sil_per_group]

        if scale:
            # scale s.t. highest number is optimal
            sil_per_group = [1 - i for i in sil_per_group]

        sil_all = sil_all.append(
            pd.DataFrame({
                'group': [group] * len(sil_per_group),
                'silhouette_score': sil_per_group
            })
        )

    sil_all = sil_all.reset_index(drop=True)
    sil_means = sil_all.groupby('group').mean()
    asw = sil_means['silhouette_score'].mean()

    if verbose:
        print(f'mean silhouette per cell: {sil_means}')

    if return_all:
        return asw, sil_means, sil_all

    return asw