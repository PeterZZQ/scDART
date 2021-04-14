import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import loss
from torch.autograd import Variable
from torch import autograd



def _train_mmd(model, opts, b_x_rna, b_x_atac, reg_mtx, dist_atac, dist_rna, 
               anchor_idx_rna = None, anchor_idx_atac = None, reg_anchor = 1,
               reg_d = 1, reg_g = 1, reg_mmd = 1, norm = "l1", mode = "kl"):
        
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
        loss_anchor = torch.tensor(0.0)
    loss_total.backward()

    opts["gene_act"].step()
    opts["encoder"].step()

    losses = {"loss_mmd": loss_mmd, "loss_d_atac": loss_d_atac, "loss_genact": loss_genact, "loss_d_rna": loss_d_rna, "loss_anchor": loss_anchor} 
    return losses




def match_latent(model, opts, dist_atac, dist_rna, data_loader_rna, data_loader_atac, 
                 n_epochs, reg_mtx, reg_d = 1, reg_g = 1, reg_mmd = 1, use_anchor = False, 
                 norm = "l1", mode = "kl"):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
                reg_anchor = 1, reg_d = reg_d, reg_g = reg_g, reg_mmd = reg_mmd, norm = "l1", mode = mode)
            
            else:
                losses = _train_mmd(model, opts, b_x_rna, b_x_atac, reg_mtx, b_dist_atac, b_dist_rna, reg_d = reg_d, 
                        reg_g = reg_g, reg_mmd = reg_mmd, norm = "l1", mode = mode)


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

        if epoch % 10 == 0:
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


def train_gact(model, opts, data_loader_rna, data_loader_atac, z_rna, z_atac, n_epochs, reg_mtx, 
               reg_g = 1, use_anchor = False, norm = "l1"):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reg_mtx = reg_mtx.T.type(torch.bool)
    batch_size = data_loader_rna.batch_size

    for epoch in range(n_epochs):
        ave_loss_atac = 0
        ave_loss_rna = 0
        ave_loss_genact = 0

        for data in zip(data_loader_atac, data_loader_rna):
            # data batch
            data_atac, data_rna = data

            opts["decoder"].zero_grad()
            opts["gene_act_t"].zero_grad()
            
            b_x_atac = data_atac["count"].to(device)
            b_x_rna = data_rna['count'].to(device)

            b_idx_atac = data_atac["index"].to(device)
            b_idx_rna = data_rna["index"].to(device)

            b_z_atac = z_atac[b_idx_atac,:]
            b_z_rna = z_rna[b_idx_rna,:]
            
            # the last batch is not full size
            if (min(b_x_atac.shape[0], b_x_rna.shape[0]) < batch_size): 
                continue
            
            b_x_r_rna = model["decoder"](b_z_rna)
            b_x_r_atac = model["gene_act_t"](model["decoder"](b_z_atac))
            loss_r_rna = loss.recon_loss(b_x_r_rna, b_x_rna, recon_mode = "original")
            loss_r_atac = loss.recon_loss(b_x_r_atac, b_x_atac, recon_mode = "original")
            loss_genact = reg_g * loss.pinfo_loss(model["gene_act_t"], ~reg_mtx, norm = norm)
            loss_total = loss_r_atac + loss_r_rna + loss_genact
            
            loss_total.backward()
            opts["decoder"].step()
            opts["gene_act_t"].step()

            ave_loss_atac += loss_r_atac
            ave_loss_rna += loss_r_rna
            ave_loss_genact += loss_genact

        ave_loss_atac /= min(len(data_loader_atac), len(data_loader_rna))
        ave_loss_rna /= min(len(data_loader_atac), len(data_loader_rna))
        ave_loss_genact /= min(len(data_loader_atac), len(data_loader_rna))

        if epoch % 10 == 0:
            info = [
                'ATAC recon loss: {:.5f}'.format(ave_loss_atac.item()),
                'RNA recon loss: {:.5f}'.format(ave_loss_rna.item()),
                'gene activity loss: {:.5f}'.format(ave_loss_genact.item())
            ]


            print("epoch: ", epoch)
            for i in info:
                print("\t", i)


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

