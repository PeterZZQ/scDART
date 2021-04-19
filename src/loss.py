import torch
import numpy as np
import torch.nn.functional as F


def compute_pairwise_distances(x, y):
    x_norm = (x**2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


def _gaussian_kernel_matrix(x, y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sigmas = torch.FloatTensor([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]).to(device)
    dist = compute_pairwise_distances(x, y)
    beta = 1. / (2. * sigmas[:,None])
    s = - beta.mm(dist.reshape((1, -1)) )
    result =  torch.sum(torch.exp(s), dim = 0)
    return result


def maximum_mean_discrepancy(x, y): #Function to calculate MMD value
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cost = torch.mean(_gaussian_kernel_matrix(x, x))
    cost += torch.mean(_gaussian_kernel_matrix(y, y))
    cost -= 2.0 * torch.mean(_gaussian_kernel_matrix(x, y))
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


# def recon_loss(recon_x, x, recon_mode = "original"):

#     if recon_mode == "original":
#         loss_recon = F.mse_loss(recon_x, x)
#     elif recon_mode == "relative":
#         mean_recon = torch.mean(recon_x, dim = 0)
#         var_recon = torch.var(recon_x, dim = 0)
#         mean_x = torch.mean(x, dim = 0)
#         var_x = torch.var(x, dim = 0)
#         # relative loss
#         loss_recon = F.mse_loss(torch.div(torch.add(x, -1.0 * mean_x), (torch.sqrt(var_x + 1e-12)+1e-12)), torch.div(torch.add(x, -1.0 * mean_recon), (torch.sqrt(var_recon + 1e-12)+1e-12)))
#     elif recon_mode == "binary":
#         loss_recon = F.binary_cross_entropy(input = recon_x, target = x)
    
#     else:
#         raise ValueError("recon_mode can only be original or relative")
    
#     return loss_recon
