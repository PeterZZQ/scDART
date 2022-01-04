import numpy as np
import pandas as pd
import warnings
import scipy
from sklearn.decomposition import PCA
import time
import torch
from sklearn.metrics import pairwise_distances

warnings.filterwarnings('ignore')

# def _pairwise_distances(x, y = None):
#     # o(n*d)
#     x_norm = (x**2).sum(1).view(-1, 1)
#     # calculate the pairwise distance between two datasets
#     if y is not None:
#         y_t = torch.transpose(y, 0, 1)
#         y_norm = (y**2).sum(1).view(1, -1)
#     else:
#         y_t = torch.transpose(x, 0, 1)
#         y_norm = x_norm.view(1, -1)
    
#     # o(n*d) + o(n*d) + o(n*n*d)
#     dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
#     # Ensure diagonal is zero if x=y
#     if y is None:
#         dist = dist - torch.diag(dist.diag)

#     return torch.clamp(dist, 0.0, np.inf)

def lsi_ATAC(X, k = 100, use_first = False):
    """\
    Description:
    ------------
        Compute LSI with TF-IDF transform, i.e. SVD on document matrix, can do tsne on the reduced dimension

    Parameters:
    ------------
        X: cell by feature(region) count matrix
        k: number of latent dimensions
        use_first: since we know that the first LSI dimension is related to sequencing depth, we just ignore the first dimension since, and only pass the 2nd dimension and onwards for t-SNE
    
    Returns:
    -----------
        latent: cell latent matrix
    """    
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.decomposition import TruncatedSVD

    # binarize the scATAC-Seq count matrix
    bin_X = np.where(X < 1, 0, 1)
    
    # perform Latent Semantic Indexing Analysis
    # get TF-IDF matrix
    tfidf = TfidfTransformer(norm='l2', sublinear_tf=True)
    normed_count = tfidf.fit_transform(bin_X)

    # perform SVD on the sparse matrix
    lsi = TruncatedSVD(n_components = k, random_state=42)
    lsi_r = lsi.fit_transform(normed_count)
    
    # use the first component or not
    if use_first:
        return lsi_r
    else:
        return lsi_r[:, 1:]

def quantile_norm(dist_mtx, reference, replace = False):
    # sampling and don't put back
    reference = np.sort(np.random.choice(reference.reshape(-1), dist_mtx.shape[0] * dist_mtx.shape[1], replace = replace))
    dist_temp = dist_mtx.reshape(-1)
    dist_idx = np.argsort(dist_temp)
    dist_temp[dist_idx] = reference
    return dist_temp.reshape(dist_mtx.shape[0], dist_mtx.shape[1])

def phate_similarity(data, n_neigh = 5, t = 5, use_potential = True, n_pca = 100, num_anchor = None, method = "exact", **kwargs):
    """\
    Description:
    ------------
        Calculate diffusion distance using Phate/Diffusion Map method
    
    Parameters:
    ------------
        data: 
            Feature matrix of dimension (n_samples, n_features)
        n_neigh:
            The number of neighbor in knn for graph construction
        t:
            The transition timestep t
        use_potential:
            Using potential distance or not, if use, the same as Phate; if not, the same as diffusion map

    Returns:
    -----------    
        dist:
            Similarity matrix
    """
    import graphtools as gt
    from scipy.spatial.distance import pdist, squareform
    from sklearn.neighbors import NearestNeighbors
    if method == "exact":
        start = time.time()
        G = gt.Graph(data, n_pca = n_pca, knn = n_neigh, **kwargs)
        T = G.diff_op
        if scipy.sparse.issparse(T):
            T = T.toarray()

        T_t = np.linalg.matrix_power(T, t)
        if use_potential:
            U_t = - np.log(T_t + 1e-7)
        else:
            U_t = T_t
        # calculate pairwise feature vector distance
        dist = squareform(pdist(U_t))
        end = time.time()
    else:
        start = time.time()
        
        anchor_idx = np.array([False] * data.shape[0])
        if num_anchor is not None:
            # randomly choose num_anchor of cells as the anchor cells for distance calculation
            anchor_idx[np.random.choice(data.shape[0], num_anchor, replace = False)] = True
        else:
            # if num_anchor is not given, then subsample the data matrix by 10.
            anchor_idx[::10] = True
        
        data_anchor = data[anchor_idx,:]
        G_anchor = gt.Graph(data_anchor, n_pca = n_pca, knn = n_neigh, **kwargs)
        T_anchor = G_anchor.diff_op
        if scipy.sparse.issparse(T_anchor):
            T_anchor = T_anchor.toarray()

        T_anchor_t = np.linalg.matrix_power(T_anchor, t)
        if use_potential:
            U_anchor_t = - np.log(T_anchor_t + 1e-7)
        else:
            U_anchor_t = T_anchor_t

        # mutual nearest neighbors, o(n*n) -> o(n*k), k is the number of anchor nodes
        dist = pairwise_distances(X = data[~anchor_idx,:], Y = data_anchor)
        knn_index = np.argpartition(dist, kth = n_neigh - 1, axis = 1)[:,(n_neigh-1)]
        kth_dist = np.take_along_axis(dist, knn_index[:,None], axis = 1)
        K = dist/kth_dist 
        K = (dist <= kth_dist) * np.exp(-K) 
        K = K/np.sum(K, axis = 1)[:,None]
        U_query_t = np.matmul(K, U_anchor_t)
        
        # features, cannot deal with distance matrix directly, or the distances between query nodes are unknown.
        U_t = np.zeros((data.shape[0], U_anchor_t.shape[1]))
        U_t[anchor_idx,:] = U_anchor_t
        U_t[~anchor_idx,:] = U_query_t

        dist = squareform(pdist(U_t))
        end = time.time()
    
    print("running time(sec):", end-start)
    return dist


def diffu_distance(data, n_neigh = 5, ts = [30,40,50,60], use_potential = False, dr = "lsi", n_components = 100, method = "exact", n_anchor = None, **kwargs):
    diffu = np.zeros((data.shape[0], data.shape[0]))
    if dr == "lsi":
        data = lsi_ATAC(data, k = n_components, use_first = False)
    elif dr == "pca":
        data = PCA(n_components = n_components).fit_transform(data)

    for t in ts:
        diffu_t = phate_similarity(data, n_neigh = n_neigh, t = t, use_potential = use_potential, method = method, n_pca = None, num_anchor = n_anchor, **kwargs)
        diffu_t = diffu_t/np.linalg.norm(diffu_t)
        diffu += diffu_t
    # average
    diffu = diffu/len(ts)
    return diffu


#-----------------------------------------------------------------------------------------
#
#   With exampler
#
#-----------------------------------------------------------------------------------------

def phate_similarity_exampler(data, data_exampler, n_neigh = 5, t = 5, use_potential = True, n_pca = 100, n_exampler = None, method = "exact", **kwargs):
    """\
    Description:
    ------------
        Calculate diffusion distance using Phate/Diffusion Map method
    
    Parameters:
    ------------
        data: 
            Feature matrix of dimension (n_samples, n_features)
        n_neigh:
            The number of neighbor in knn for graph construction
        t:
            The transition timestep t
        use_potential:
            Using potential distance or not, if use, the same as Phate; if not, the same as diffusion map

    Returns:
    -----------    
        dist:
            Similarity matrix
    """
    import graphtools as gt
    from scipy.spatial.distance import pdist, squareform
    from sklearn.neighbors import NearestNeighbors
    from sklearn.cluster import KMeans

    start = time.time()

    # calculate the exampler
    G_exampler = gt.Graph(data_exampler, n_pca = n_pca, knn = n_neigh, **kwargs)
    T_exampler = G_exampler.diff_op
    if scipy.sparse.issparse(T_exampler):
        T_exampler = T_exampler.toarray()

    T_exampler_t = np.linalg.matrix_power(T_exampler, t)
    if use_potential:
        U_exampler_t = - np.log(T_exampler_t + 1e-7)
    else:
        U_exampler_t = T_exampler_t
    
    dist_exampler = squareform(pdist(U_exampler_t))
    # calculate distance between data and exampler, choice 1: euclidean distance, choice 2: diffusion distance
    # choice 1
    # dist = pairwise_distances(X = data, Y = data_exampler)

    # choice 2
    dist = pairwise_distances(X = data, Y = data_exampler)
    knn_index = np.argpartition(dist, kth = n_neigh - 1, axis = 1)[:,(n_neigh-1)]
    kth_dist = np.take_along_axis(dist, knn_index[:,None], axis = 1)
    K = dist/kth_dist 
    K = (dist <= kth_dist) * np.exp(-K) 
    K = K/np.sum(K, axis = 1)[:,None]
    U_query_t = np.matmul(K, U_exampler_t)
    # features, cannot deal with distance matrix directly, or the distances between query nodes are unknown.
    dist = pairwise_distances(X = U_query_t, Y = U_exampler_t)
    end = time.time()
    
    print("running time(sec):", end-start)
    return dist, dist_exampler


def diffu_distance_exampler(data, n_neigh = 5, ts = [30,40,50,60], use_potential = False, dr = "lsi", n_components = 100, method = "exact", n_exampler = None, **kwargs):
    from sklearn.cluster import KMeans
    if n_exampler is None:
        # number of exampler is set to 0.1 * datasize
        n_exampler = int(0.1 * data.shape[0])
    dist = np.zeros((data.shape[0], n_exampler))
    dist_exampler = np.zeros((n_exampler, n_exampler))

    if dr == "lsi":
        data_pca = lsi_ATAC(data, k = n_components, use_first = False)
    elif dr == "pca":
        data_pca = PCA(n_components = n_components).fit_transform(data)

    kmeans = KMeans(n_clusters = n_exampler, init = "k-means++", random_state = 0).fit(data_pca)
    groups = kmeans.labels_
    data_exampler = np.concatenate([np.mean(data[groups == group,:], axis = 0, keepdims = True) for group in np.sort(np.unique(groups))], axis = 0)
    data_exampler_pca = np.concatenate([np.mean(data_pca[groups == group,:], axis = 0, keepdims = True) for group in np.sort(np.unique(groups))], axis = 0)

    for t in ts:
        # for each run, the exampler is the same (defined by random_state)
        dist_t, dist_exampler_t = phate_similarity_exampler(data_pca, data_exampler_pca, n_neigh = n_neigh, t = t, use_potential = use_potential, method = method, n_pca = None, n_exampler = n_exampler, **kwargs)
        dist += dist_t/np.linalg.norm(dist_t)
        dist_exampler += dist_exampler_t/np.linalg.norm(dist_exampler_t)
    # average
    dist = dist/len(ts)
    dist_exampler = dist_exampler/len(ts)
    return dist, dist_exampler, data_exampler