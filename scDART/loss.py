import torch
import numpy as np
import torch.nn.functional as F


# def pairwise_distance_old(x):
#     """\
#     Description:
#     -----------
#         Pytorch implementation of pairwise distance, similar to squareform(pdist(x))
        
#     Parameters:
#     -----------
#         x: sample by feature matrix
#     Returns:
#     -----------
#         dist: sample by sample pairwise distance
#     """
#     x_norm = (x**2).sum(1).view(-1, 1)
#     y_norm = x_norm.view(1, -1)
#     dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(x, 0, 1))
#     dist = torch.sqrt(dist + 1e-2)
#     return dist 

def compute_pairwise_distances(x, y):
    x_norm = (x**2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


def _gaussian_kernel_matrix(x, y, device):    
    sigmas = torch.FloatTensor([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]).to(device)
    dist = compute_pairwise_distances(x, y)
    beta = 1. / (2. * sigmas[:,None])
    s = - beta.mm(dist.reshape((1, -1)) )
    result =  torch.sum(torch.exp(s), dim = 0)
    return result


def _maximum_mean_discrepancy(x, y, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')): #Function to calculate MMD value    
    cost = torch.mean(_gaussian_kernel_matrix(x, x, device))
    cost += torch.mean(_gaussian_kernel_matrix(y, y, device))
    cost -= 2.0 * torch.mean(_gaussian_kernel_matrix(x, y, device))
    cost = torch.sqrt(cost ** 2 + 1e-9)
    if cost.data.item()<0:
        cost = torch.FloatTensor([0.0]).to(device)

    return cost

def maximum_mean_discrepancy(xs, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')): #Function to calculate MMD value
    nbatches = len(xs)
    ref_batch = 0
    # assuming batch 0 is the reference batch
    cost = 0
    # within batch
    for batch in range(nbatches):
        if batch == ref_batch:
            cost += (nbatches - 1) * torch.mean(_gaussian_kernel_matrix(xs[batch], xs[batch], device))
        else:
            cost += torch.mean(_gaussian_kernel_matrix(xs[batch], xs[batch], device))
    
    # between batches
    for batch in range(1, nbatches):
        cost -= 2.0 * torch.mean(_gaussian_kernel_matrix(xs[ref_batch], xs[batch], device))
    
    cost = torch.sqrt(cost ** 2 + 1e-9)
    if cost.data.item()<0:
        cost = torch.FloatTensor([0.0]).to(device)

    return cost

# Loss
def pinfo_loss(model, mask, norm = "l2"):
    W = None
    for _, layers in model.fc_layers.named_children():
        for name, layer in layers.named_children():
            if name[:3] == "lin":
                if W is None:
                    W = layer.weight
                else:
                    W = torch.mm(layer.weight,W)
    if norm == "l2":
        loss = torch.norm(mask.T * W, p = "fro")
    else:
        # l1 norm
        loss = torch.sum(torch.abs(mask.T * W))
    return loss


# def dist_loss(z, diff_sim, dist_mode = "mse"):
#     # cosine similarity loss
#     latent_sim = pairwise_distance(z)

#     if dist_mode == "inner_product":
#         # normalize latent similarity matrix
#         latent_sim = latent_sim / torch.norm(latent_sim, p='fro')
#         # unnecessary to normalize diff_sim, diff_sim fixed
#         diff_sim = diff_sim / torch.norm(diff_sim, p = 'fro')

#         # inner product loss, maximize, so add negative before, in addition, make sure those two values are normalized, with norm 1
#         loss_dist = - torch.sum(diff_sim * latent_sim) 

#     elif dist_mode == "mse": 
#         # MSE loss
#         # normalize latent similarity matrix
#         latent_sim = latent_sim / torch.norm(latent_sim, p='fro')
#         diff_sim = diff_sim / torch.norm(diff_sim, p = 'fro')

#         loss_dist = torch.norm(diff_sim - latent_sim, p = 'fro')
    
#     elif dist_mode == "similarity":
#         diff_sim = diff_sim / torch.sum(diff_sim)
        
#         # using t-student instead of Gaussian to circumvent the crowding problem
#         latent_t = 1 / (1 + latent_sim ** 2)
#         # normalize for the same scale
#         latent_t = latent_t / torch.sum(latent_t)
#         # symmetric KL -> JS distance, use KL to ensure that close distance in high dimensional space is also close in low dimensional space, 
#         # but large distance in high dimensional space is not necessarily large in low dimensional space, thus enforce local
#         # we want constraint on global, thus JS
#         M = 0.5 * (diff_sim + latent_t)
#         loss_dist = torch.sum(diff_sim * torch.log(diff_sim/M) + latent_t * torch.log(latent_t/M))

#     else:
#         raise ValueError("`dist_model` should only be `mse` or `inner_product`")

#     return loss_dist

# def pearson(z, diff_sim):
#     latent_sim = torch.mm(z, z.T)   
#     var_latent_sim = latent_sim - torch.mean(latent_sim)
#     var_diff_sim = (diff_sim - torch.mean(diff_sim))

#     score = torch.sum(var_latent_sim * var_diff_sim) / (torch.sqrt(torch.sum(var_latent_sim ** 2)) * torch.sqrt(torch.sum(var_diff_sim ** 2)))

#     return - score


def dist_loss(z, diff_sim, mask = None, mode = "mse"):
    # cosine similarity loss
    latent_sim = compute_pairwise_distances(z, z)
    if mode == "mse":
        latent_sim = latent_sim / torch.norm(latent_sim, p='fro')
        diff_sim = diff_sim / torch.norm(diff_sim, p = 'fro')

        if mask is not None:
            loss_dist = torch.norm((diff_sim - latent_sim) * mask, p = 'fro')
        else:   
            loss_dist = torch.norm(diff_sim - latent_sim, p = 'fro')
    
    elif mode == "kl":
        # latent_sim = 1/(1 + latent_sim)
        # diff_sim = 1/(1 + diff_sim)
        Q_dist = latent_sim / torch.sum(latent_sim) + 1e-12
        P_dist = diff_sim / torch.sum(diff_sim) + 1e-12
        loss_dist = torch.sum(Q_dist * torch.log(Q_dist / P_dist))
    return loss_dist


# def sim_loss(z, sim_x):
#     sim_z = torch.mm(z, z.T)
#     sim_z = sim_z/torch.norm(sim_z)
#     # when the norm of z is constant, using inner product the same as mse
#     loss_dist = - torch.sum(sim_z * sim_x) 
#     return loss_dist


# def sim_loss_tsne(z, sim_x):
#     sim_x = sim_x/torch.sum(sim_x)

#     latent_sim = pairwise_distance(z)
#     latent_sim = 1 / (1 + latent_sim ** 2)
#     latent_sim = latent_sim / torch.sum(latent_sim)

#     loss_kl = torch.sum(sim_x * torch.log(sim_x/latent_sim))
#     return loss_kl

def recon_loss(recon_x, x, recon_mode = "original"):

    if recon_mode == "original":
        loss_recon = F.mse_loss(recon_x, x)
    elif recon_mode == "relative":
        mean_recon = torch.mean(recon_x, dim = 0)
        var_recon = torch.var(recon_x, dim = 0)
        mean_x = torch.mean(x, dim = 0)
        var_x = torch.var(x, dim = 0)
        # relative loss
        loss_recon = F.mse_loss(torch.div(torch.add(x, -1.0 * mean_x), (torch.sqrt(var_x + 1e-12)+1e-12)), torch.div(torch.add(x, -1.0 * mean_recon), (torch.sqrt(var_recon + 1e-12)+1e-12)))
    elif recon_mode == "binary":
        loss_recon = F.binary_cross_entropy(input = recon_x, target = x)
    
    else:
        raise ValueError("recon_mode can only be original or relative")
    
    return loss_recon
