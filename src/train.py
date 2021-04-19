import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import src.loss as loss
from torch.autograd import Variable
from torch import autograd
import src.diffusion_dist as diff
import src.model as model
import src.post_align as palign

def scDART_train(EMBED_CONFIG, reg_mtx, train_rna_loader, train_atac_loader, test_rna_loader, test_atac_loader, \
    n_epochs = 1001, use_anchor = True, n_anchor = None, ts = None, reg_d = 1, reg_g = 1, reg_mmd = 1, \
        l_dist_type = 'kl', device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        
        """\
            Train scDART model
        """

        print("Device: ", device)

        #TODO: check parameters, fixed or hyperparamters
        # calculate the distance

        # n_anchor here is the number of anchor nodes used for distance calculation, if n_anchor is none, then use exact way of calculating the distance
        # if n_anchor is given, then use fast way
        if n_anchor == None:
            method = "exact"
            print("Number of anchor cells not specified, using exact mode for distance calculation instead.")
        else:
            method = "fast"
            print("Using fast mode for distance calculation. Number of anchor cells:.{:d}".format(n_anchor))
        
        for data in test_rna_loader:
            dist_rna = diff.diffu_distance(data["count"].numpy(), ts = ts, 
            use_potential = False, dr = "pca", method = method , n_anchor = n_anchor)

        for data in test_atac_loader:
            dist_atac = diff.diffu_distance(data["count"].numpy(), ts = ts, 
            use_potential = False, dr = "lsi", method = "exact", n_anchor = n_anchor)

        dist_rna = dist_rna/np.linalg.norm(dist_rna)
        dist_atac = dist_atac/np.linalg.norm(dist_atac)
        dist_rna = torch.FloatTensor(dist_rna).to(device)
        dist_atac = torch.FloatTensor(dist_atac).to(device)

        genact = model.gene_act(features = EMBED_CONFIG["gact_layers"], dropout_rate = 0.0, negative_slope = 0.2).to(device)
        encoder = model.Encoder(features = EMBED_CONFIG["proj_layers"], dropout_rate = 0.0, negative_slope = 0.2).to(device)
        decoder = model.Decoder(features = EMBED_CONFIG["proj_layers"][::-1], dropout_rate = 0.0, negative_slope = 0.2).to(device)
        genact_t = model.gene_act_t(features = EMBED_CONFIG["gact_layers"][::-1], dropout_rate = 0.0, negative_slope = 0.2).to(device)
        model_dict = {"gene_act": genact, "encoder": encoder, "decoder": decoder, "gene_act_t": genact_t}

        learning_rate = EMBED_CONFIG['learning_rate']
        opt_genact = torch.optim.Adam(genact.parameters(), lr = learning_rate)
        opt_encoder = torch.optim.Adam(encoder.parameters(), lr = learning_rate)
        opt_decoder = torch.optim.Adam(decoder.parameters(), lr = learning_rate)
        opt_genact_t = torch.optim.Adam(genact_t.parameters(), lr = learning_rate)
        opt_dict = {"gene_act": opt_genact, "encoder": opt_encoder, "decoder": opt_decoder, "gene_act_t": opt_genact_t}

        print("Model:", model_dict)

        match_latent(model = model_dict, opts = opt_dict, dist_atac = dist_atac, dist_rna = dist_rna, 
                            data_loader_rna = train_rna_loader, data_loader_atac = train_atac_loader, n_epochs = n_epochs, 
                            reg_mtx = reg_mtx, reg_d = reg_d, reg_g = reg_g, reg_mmd = reg_mmd, use_anchor = use_anchor, norm = "l1", 
                            mode = l_dist_type)

        # with torch.no_grad():
        #     for data in test_rna_loader:
        #         z_rna = model_dict["encoder"](data['count'].to(device)).cpu().detach()

        #     for data in test_atac_loader:
        #         z_atac = model_dict["encoder"](model_dict["gene_act"](data['count'].to(device))).cpu().detach()

        # # post-maching
        # z_rna, z_atac = palign.match_alignment(z_rna = z_rna, z_atac = z_atac, k = k)
        # z_atac, z_rna = palign.match_alignment(z_rna = z_atac, z_atac = z_rna, k = k)
    
        return model_dict


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

