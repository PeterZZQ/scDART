import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import loss
from torch.autograd import Variable
from torch import autograd



def _train_mmd(model, opts, b_x_rna, b_x_atac, reg_mtx, dist_atac, dist_rna, 
               anchor_idx_rna, anchor_idx_atac, reg_anchor,
               reg_d, reg_g, reg_mmd, norm, mode, device):
    """\
    Description:
    ------------
        training model on one batch of data
    Parameter
    ------------
        model: neural network model
        opts: dictionary of optimizer
        b_x_rna: one batch of scRNA-Seq
        b_x_atac: one batch of scATAC-Seq
        reg_mtx: gene activity matrix
        dist_atac: diffusion distance matrix of scATAC-Seq
        dist_rna: diffusion distance matrix of scRNA-Seq
        anchor_idx_rna: matching anchor for RNA
        anchor_idx_atac: matching anchor for ATAC
    """        
    # train generator
    opts["gene_act"].zero_grad()
    opts["encoder"].zero_grad()

    # latent space generated from atac data
    b_z_atac = model["encoder"](model["gene_act"](b_x_atac))
    b_z_rna = model["encoder"](b_x_rna)

    # regularization term
    loss_d_atac = reg_d * loss.dist_loss(z = b_z_atac, diff_sim = dist_atac, mode = mode)
    loss_d_rna = reg_d * loss.dist_loss(z = b_z_rna, diff_sim = dist_rna, mode = mode)

    loss_genact = reg_g * loss.pinfo_loss(model["gene_act"], ~reg_mtx, norm = norm)
    loss_mmd = reg_mmd * loss.maximum_mean_discrepancy(b_z_atac, b_z_rna)

    loss_total = loss_d_atac + loss_genact + loss_d_rna + loss_mmd
    if anchor_idx_rna is not None and anchor_idx_atac is not None:
        loss_anchor = reg_anchor * torch.norm(torch.mean(b_z_atac[anchor_idx_atac,:], dim = 0) - torch.mean(b_z_rna[anchor_idx_rna,:], dim = 0))
        loss_total += loss_anchor
    else:
        loss_anchor = torch.tensor(0.0).to(device)
    loss_total.backward()

    opts["gene_act"].step()
    opts["encoder"].step()

    losses = {"loss_mmd": loss_mmd, "loss_d_atac": loss_d_atac, "loss_genact": loss_genact, "loss_d_rna": loss_d_rna, "loss_anchor": loss_anchor} 
    return losses




def match_latent(model, opts, dist_atac, dist_rna, data_loader_rna, data_loader_atac, 
                 n_epochs, reg_mtx, reg_d = 1, reg_g = 1, reg_mmd = 1, use_anchor = False, 
                 norm = "l1", mode = "kl", device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

    reg_mtx = reg_mtx.type(torch.bool)
    batch_size = data_loader_rna.batch_size

    loss_record = []

    for epoch in range(n_epochs):
        ave_loss_mmd = 0
        ave_loss_d_atac = 0
        ave_loss_genact = 0
        ave_loss_d_rna = 0
        ave_loss_anchor = 0

        for data in zip(data_loader_atac, data_loader_rna):
            # data batch
            data_atac, data_rna = data
            
            b_idx_atac = data_atac["index"].to(device)
            b_dist_atac = dist_atac[b_idx_atac,:][:,b_idx_atac]
            b_x_atac = data_atac["count"].to(device)

            b_x_rna = data_rna['count'].to(device)
            b_idx_rna = data_rna["index"].to(device)
            b_dist_rna = dist_rna[b_idx_rna,:][:,b_idx_rna]

            # the last batch is not full size
            if (min(b_x_atac.shape[0], b_x_rna.shape[0]) < batch_size): 
                continue
                
            b_anchor_rna = data_rna["is_anchor"].to(device)   
            b_anchor_atac = data_atac["is_anchor"].to(device)
            # do not allow batch size 1
            if use_anchor and (min(b_x_atac[b_anchor_atac,:].shape[0], b_x_rna[b_anchor_rna,:].shape[0]) > 1):
                losses = _train_mmd(model, opts, b_x_rna, b_x_atac, reg_mtx, b_dist_atac, b_dist_rna, anchor_idx_atac = b_anchor_atac, anchor_idx_rna = b_anchor_rna, 
                reg_anchor = 1, reg_d = reg_d, reg_g = reg_g, reg_mmd = reg_mmd, norm = norm, mode = mode, device = device)
            
            else:
                losses = _train_mmd(model, opts, b_x_rna, b_x_atac, reg_mtx, b_dist_atac, b_dist_rna, anchor_idx_atac = None, anchor_idx_rna = None,
                reg_anchor = 0, reg_d = reg_d, reg_g = reg_g, reg_mmd = reg_mmd, norm = norm, mode = mode, device = device)


            ave_loss_mmd += losses["loss_mmd"]
            ave_loss_d_atac += losses["loss_d_atac"]
            ave_loss_genact += losses["loss_genact"]  
            ave_loss_d_rna += losses["loss_d_rna"]
            ave_loss_anchor += losses["loss_anchor"]          

        ave_loss_mmd /= min(len(data_loader_atac), len(data_loader_rna))
        ave_loss_d_atac /= min(len(data_loader_atac), len(data_loader_rna))
        ave_loss_genact /= min(len(data_loader_atac), len(data_loader_rna))
        ave_loss_d_rna /= min(len(data_loader_atac), len(data_loader_rna))
        ave_loss_anchor /= min(len(data_loader_atac), len(data_loader_rna))

        loss_record.append({
            "loss_mmd": ave_loss_mmd.item(),
            "loss_d_atac": ave_loss_d_atac.item(),
            "loss_d_rna": ave_loss_d_rna.item(),
            "gene activity loss": ave_loss_genact.item()
        })
        
        if epoch % 100 == 0:
            info = [
                'mmd loss: {:.3f}'.format(ave_loss_mmd.item()),
                'ATAC dist loss: {:.3f}'.format(ave_loss_d_atac.item()),
                'RNA dist loss: {:.3f}'.format(ave_loss_d_rna.item()),
                'gene activity loss: {:.3f}'.format(ave_loss_genact.item()),
                'anchor matching loss: {:.3f}'.format(ave_loss_anchor.item())
            ]


            print("epoch: ", epoch)
            for i in info:
                print("\t", i)

    return loss_record


def infer_gact(model, mask, thresh = None):
    W = None

    for _, layers in model.fc_layers.named_children():
        for name, layer in layers.named_children():
            if name[:3] == "lin":
                if W is None:
                    W = layer.weight
                else:
                    W = torch.mm(layer.weight,W)
    
    if W.shape[0] == mask.shape[0]:
        gact = torch.abs(mask * W)
    else:
        gact = torch.abs(mask * W.T)
    
    if thresh is None:
        thresh = 1
    
    thresh = thresh * (torch.max(gact) + torch.min(gact))/2
    gact = (gact >= thresh)

    return gact

#-----------------------------------------------------------------------------------------
#
#   fix the projection module, better learn the transformation
#
#-----------------------------------------------------------------------------------------
def _train_mmd_fix(model, opts, b_x_rna, b_x_atac, reg_mtx, dist_atac, dist_rna, 
                  anchor_idx_rna, anchor_idx_atac, reg_anchor,
                  reg_d, reg_g, reg_mmd, norm, mode, device):
    """\
    Description:
    ------------
        training model on one batch of data
    Parameter
    ------------
        model: neural network model
        opts: dictionary of optimizer
        b_x_rna: one batch of scRNA-Seq
        b_x_atac: one batch of scATAC-Seq
        reg_mtx: gene activity matrix
        dist_atac: diffusion distance matrix of scATAC-Seq
        dist_rna: diffusion distance matrix of scRNA-Seq
        anchor_idx_rna: matching anchor for RNA
        anchor_idx_atac: matching anchor for ATAC
    """        
    # train generator
    opts["gene_act"].zero_grad()
    # opts["encoder"].zero_grad()
    batch_rna = torch.FloatTensor([0]).to(device)
    batch_atac = torch.FloatTensor([1]).to(device)

    # latent space generated from atac data
    b_z_atac = model["encoder_batch"](model["gene_act"](b_x_atac), batch_atac)
    b_z_rna = model["encoder_batch"](b_x_rna, batch_rna)

    # regularization term
    loss_d_atac = reg_d * loss.dist_loss(z = b_z_atac, diff_sim = dist_atac, mode = mode)
    loss_d_rna = reg_d * loss.dist_loss(z = b_z_rna, diff_sim = dist_rna, mode = mode)

    loss_genact = reg_g * loss.pinfo_loss(model["gene_act"], ~reg_mtx, norm = norm)
    loss_mmd = reg_mmd * loss.maximum_mean_discrepancy(b_z_atac, b_z_rna)

    loss_total = loss_d_atac + loss_genact + loss_d_rna + loss_mmd
    if anchor_idx_rna is not None and anchor_idx_atac is not None:
        loss_anchor = reg_anchor * torch.norm(torch.mean(b_z_atac[anchor_idx_atac,:], dim = 0) - torch.mean(b_z_rna[anchor_idx_rna,:], dim = 0))
        loss_total += loss_anchor
    else:
        loss_anchor = torch.tensor(0.0)
    loss_total.backward()

    opts["gene_act"].step()
    # opts["encoder"].step()

    losses = {"loss_mmd": loss_mmd, "loss_d_atac": loss_d_atac, "loss_genact": loss_genact, "loss_d_rna": loss_d_rna, "loss_anchor": loss_anchor} 
    return losses



def match_latent_fix(model, opts, dist_atac, dist_rna, data_loader_rna, data_loader_atac, 
                    n_epochs, reg_mtx, reg_d = 1, reg_g = 1, reg_mmd = 1, use_anchor = False, 
                    norm = "l1", mode = "kl"):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    reg_mtx = reg_mtx.type(torch.bool)
    batch_size = data_loader_rna.batch_size

    # train projection module
    for epoch in range(100):
        ave_loss_d_rna = 0
        for data_rna in data_loader_rna:
            b_x_rna = data_rna['count'].to(device)
            b_idx_rna = data_rna["index"].to(device)
            b_dist_rna = dist_rna[b_idx_rna,:][:,b_idx_rna]
            if b_x_rna.shape[0] < batch_size:
                continue
            b_anchor_rna = data_rna["is_anchor"].to(device)   
            # calculate only distance loss
            b_z_rna = model["encoder_batch"](b_x_rna)
            loss_d_rna = reg_d * loss.dist_loss(z = b_z_rna, diff_sim = b_dist_rna, mode = mode)
            loss_d_rna.backward()
            opts["encoder_batch"].step()  

            ave_loss_d_rna += loss_d_rna
        
        ave_loss_d_rna /= len(data_loader_rna)

        if epoch % 100 == 0:
            info = [
                'RNA dist loss: {:.3f}'.format(ave_loss_d_rna.item()),
            ]
            print("epoch: ", epoch)
            for i in info:
                print("\t", i)            

    # train gene activity module
    # fix the gradient of the projection module, only leave out the batch layer
    for parameter in model["encoder_batch"].fc.parameters():
        parameter.requires_grad = False
    for parameter in model["encoder_batch"].fc_input.parameters():
        parameter.requires_grad = False

    for epoch in range(100, n_epochs):
        ave_loss_mmd = 0
        ave_loss_d_atac = 0
        ave_loss_genact = 0
        ave_loss_rna = 0
        ave_loss_anchor = 0

        for data in zip(data_loader_atac, data_loader_rna):
            # data batch
            data_atac, data_rna = data
            
            b_idx_atac = data_atac["index"].to(device)
            b_dist_atac = dist_atac[b_idx_atac,:][:,b_idx_atac]
            b_x_atac = data_atac["count"].to(device)

            b_x_rna = data_rna['count'].to(device)
            b_idx_rna = data_rna["index"].to(device)
            b_dist_rna = dist_rna[b_idx_rna,:][:,b_idx_rna]

            # the last batch is not full size
            if (min(b_x_atac.shape[0], b_x_rna.shape[0]) < batch_size): 
                continue
                
            b_anchor_rna = data_rna["is_anchor"].to(device)   
            b_anchor_atac = data_atac["is_anchor"].to(device)
            # do not allow batch size 1
            if use_anchor and (min(b_x_atac[b_anchor_atac,:].shape[0], b_x_rna[b_anchor_rna,:].shape[0]) > 1):
                losses = _train_mmd_fix(model, opts, b_x_rna, b_x_atac, reg_mtx, b_dist_atac, b_dist_rna, anchor_idx_atac = b_anchor_atac, anchor_idx_rna = b_anchor_rna, 
                reg_anchor = 1, reg_d = reg_d, reg_g = reg_g, reg_mmd = reg_mmd, norm = norm, mode = mode, device = device)
            
            else:
                losses = _train_mmd_fix(model, opts, b_x_rna, b_x_atac, reg_mtx, b_dist_atac, b_dist_rna, anchor_idx_atac = None, anchor_idx_rna = None,
                reg_anchor = 0, reg_d = reg_d, reg_g = reg_g, reg_mmd = reg_mmd, norm = norm, mode = mode, device = device)


            ave_loss_mmd += losses["loss_mmd"]
            ave_loss_d_atac += losses["loss_d_atac"]
            ave_loss_genact += losses["loss_genact"]  
            ave_loss_rna += losses["loss_d_rna"]
            ave_loss_anchor += losses["loss_anchor"]          

        ave_loss_mmd /= min(len(data_loader_atac), len(data_loader_rna))
        ave_loss_d_atac /= min(len(data_loader_atac), len(data_loader_rna))
        ave_loss_genact /= min(len(data_loader_atac), len(data_loader_rna))
        ave_loss_rna /= min(len(data_loader_atac), len(data_loader_rna))
        ave_loss_anchor /= min(len(data_loader_atac), len(data_loader_rna))

        if epoch % 100 == 0:
            info = [
                'mmd loss: {:.3f}'.format(ave_loss_mmd.item()),
                'ATAC dist loss: {:.3f}'.format(ave_loss_d_atac.item()),
                'RNA dist loss: {:.3f}'.format(ave_loss_rna.item()),
                'gene activity loss: {:.3f}'.format(ave_loss_genact.item()),
                'anchor matching loss: {:.3f}'.format(ave_loss_anchor.item())
            ]


            print("epoch: ", epoch)
            for i in info:
                print("\t", i)


'''
#-----------------------------------------------------------------------------------------
#
#   With exampler
#
#-----------------------------------------------------------------------------------------

def compute_pairwise_distances(x, y):
    x_norm = (x**2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)

def dist_loss(z, z_exampler, diff_sim, mask = None, mode = "mse"):
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
        Q_dist = latent_sim / torch.sum(latent_sim) + 1e-12
        P_dist = diff_sim / torch.sum(diff_sim) + 1e-12
        loss_dist = torch.sum(Q_dist * torch.log(Q_dist / P_dist))
    return loss_dist

def _train_mmd_exampler(model, opts, b_x_rna, b_x_atac, reg_mtx, dist_atac, dist_rna, dist_atac_exampler, dist_rna_exampler, exampler_atac, exampler_rna,
                       anchor_idx_rna = None, anchor_idx_atac = None, reg_anchor = 1,
                       reg_d = 1, reg_g = 1, reg_mmd = 1, norm = "l1", mode = "kl", device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        
    # train generator
    opts["gene_act"].zero_grad()
    opts["encoder"].zero_grad()

    # latent space generated from atac data
    b_z_atac = model["encoder"](model["gene_act"](b_x_atac))
    b_z_rna = model["encoder"](b_x_rna)
    b_z_atac_exampler = model["encoder"](model["gene_act"](exampler_atac))
    b_z_rna_exampler = model["encoder"](exampler_rna)

    # regularization term
    # distance loss atac
    latent_dist_atac = compute_pairwise_distances(b_z_atac, b_z_atac_exampler)
    latent_dist_rna = compute_pairwise_distances(b_z_rna, b_z_rna_exampler)
    latent_dist_atac_exampler = compute_pairwise_distances(b_z_atac_exampler, b_z_atac_exampler)
    latent_dist_rna_exampler = compute_pairwise_distances(b_z_rna_exampler, b_z_rna_exampler)

    Q_dist_atac = latent_dist_atac / torch.sum(latent_dist_atac) + 1e-12
    P_dist_atac = dist_atac / torch.sum(dist_atac) + 1e-12
    loss_d_atac = reg_d * torch.sum(Q_dist_atac * torch.log(Q_dist_atac / P_dist_atac)) 

    Q_dist_rna = latent_dist_rna / torch.sum(latent_dist_rna) + 1e-12
    P_dist_rna = dist_rna / torch.sum(dist_rna) + 1e-12
    loss_d_rna = reg_d * torch.sum(Q_dist_rna * torch.log(Q_dist_rna / P_dist_rna))   

    Q_dist_atac_exampler = latent_dist_atac_exampler / torch.sum(latent_dist_atac_exampler) + 1e-12
    P_dist_atac_exampler = dist_atac_exampler / torch.sum(dist_atac_exampler) + 1e-12
    loss_d_atac += reg_d * torch.sum(Q_dist_atac_exampler * torch.log(Q_dist_atac_exampler / P_dist_atac_exampler)) 

    Q_dist_rna_exampler = latent_dist_rna_exampler / torch.sum(latent_dist_rna_exampler) + 1e-12
    P_dist_rna_exampler = dist_rna_exampler / torch.sum(dist_rna_exampler) + 1e-12
    loss_d_rna += reg_d * torch.sum(Q_dist_rna_exampler * torch.log(Q_dist_rna_exampler / P_dist_rna_exampler))   
    
    # other losses
    loss_genact = reg_g * loss.pinfo_loss(model["gene_act"], ~reg_mtx, norm = norm)
    loss_mmd = reg_mmd * loss.maximum_mean_discrepancy(b_z_atac_exampler, b_z_rna_exampler, device = device)
    # loss_mmd += reg_mmd * loss.maximum_mean_discrepancy(b_z_atac, b_z_rna)

    loss_total = loss_d_atac + loss_genact + loss_d_rna + loss_mmd
    if anchor_idx_rna is not None and anchor_idx_atac is not None:
        loss_anchor = reg_anchor * torch.norm(torch.mean(b_z_atac[anchor_idx_atac,:], dim = 0) - torch.mean(b_z_rna[anchor_idx_rna,:], dim = 0))
        loss_total += loss_anchor
    else:
        loss_anchor = torch.tensor(0.0)
    loss_total.backward()

    opts["gene_act"].step()
    opts["encoder"].step()

    losses = {"loss_mmd": loss_mmd, "loss_d_atac": loss_d_atac, "loss_genact": loss_genact, "loss_d_rna": loss_d_rna, "loss_anchor": loss_anchor} 
    return losses


def match_latent_exampler(model, opts, dist_atac, dist_rna, dist_atac_exampler, dist_rna_exampler, exampler_atac, exampler_rna, data_loader_rna, data_loader_atac, n_epochs, reg_mtx, reg_d = 1, reg_g = 1, reg_mmd = 1, use_anchor = False, norm = "l1", mode = "kl", device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

    reg_mtx = reg_mtx.type(torch.bool)
    batch_size = data_loader_rna.batch_size

    for epoch in range(n_epochs):
        ave_loss_mmd = 0
        ave_loss_d_atac = 0
        ave_loss_genact = 0
        ave_loss_rna = 0
        ave_loss_anchor = 0

        for data in zip(data_loader_atac, data_loader_rna):
            # data batch
            data_atac, data_rna = data
            
            b_idx_atac = data_atac["index"].to(device)
            b_dist_atac = dist_atac[b_idx_atac,:]
            b_x_atac = data_atac["count"].to(device)

            b_x_rna = data_rna['count'].to(device)
            b_idx_rna = data_rna["index"].to(device)
            b_dist_rna = dist_rna[b_idx_rna,:]

            # the last batch is not full size
            if (min(b_x_atac.shape[0], b_x_rna.shape[0]) < batch_size): 
                continue
                
            b_anchor_rna = data_rna["is_anchor"].to(device)   
            b_anchor_atac = data_atac["is_anchor"].to(device)
            # do not allow batch size 1
            if use_anchor and (min(b_x_atac[b_anchor_atac,:].shape[0], b_x_rna[b_anchor_rna,:].shape[0]) > 1):
                losses = _train_mmd_exampler(model, opts, b_x_rna, b_x_atac, reg_mtx, b_dist_atac, b_dist_rna, dist_atac_exampler, dist_rna_exampler, exampler_atac, exampler_rna, anchor_idx_atac = b_anchor_atac, anchor_idx_rna = b_anchor_rna, 
                reg_anchor = 1, reg_d = reg_d, reg_g = reg_g, reg_mmd = reg_mmd, norm = "l1", mode = mode, device = device)
            
            else:
                losses = _train_mmd_exampler(model, opts, b_x_rna, b_x_atac, reg_mtx, b_dist_atac, b_dist_rna, dist_atac_exampler, dist_rna_exampler, exampler_atac, exampler_rna, reg_d = reg_d, 
                        reg_g = reg_g, reg_mmd = reg_mmd, norm = "l1", mode = mode, device = device)


            ave_loss_mmd += losses["loss_mmd"]
            ave_loss_d_atac += losses["loss_d_atac"]
            ave_loss_genact += losses["loss_genact"]  
            ave_loss_rna += losses["loss_d_rna"]
            ave_loss_anchor += losses["loss_anchor"]          

        ave_loss_mmd /= min(len(data_loader_atac), len(data_loader_rna))
        ave_loss_d_atac /= min(len(data_loader_atac), len(data_loader_rna))
        ave_loss_genact /= min(len(data_loader_atac), len(data_loader_rna))
        ave_loss_rna /= min(len(data_loader_atac), len(data_loader_rna))
        ave_loss_anchor /= min(len(data_loader_atac), len(data_loader_rna))

        if epoch % 100 == 0:
            info = [
                'mmd loss: {:.3f}'.format(ave_loss_mmd.item()),
                'ATAC dist loss: {:.3f}'.format(ave_loss_d_atac.item()),
                'RNA dist loss: {:.3f}'.format(ave_loss_rna.item()),
                'gene activity loss: {:.3f}'.format(ave_loss_genact.item()),
                'anchor matching loss: {:.3f}'.format(ave_loss_anchor.item())
            ]


            print("epoch: ", epoch)
            for i in info:
                print("\t", i)
'''