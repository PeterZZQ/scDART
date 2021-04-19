import numpy as np
import pandas as pd
import warnings
import scipy
from sklearn.decomposition import PCA
import time
import torch
from sklearn.metrics import pairwise_distances

warnings.filterwarnings('ignore')


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
    
    else:        
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
        
    return dist


def diffu_distance(data, n_neigh = 5, ts = [30,40,50,60], use_potential = False, dr = "lsi", n_components = 100, method = "exact", n_anchor = None, **kwargs):
    start = time.time()
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
    end = time.time()
    print("Diffusion distance calculated, time used (sec):", end-start)
    return diffu