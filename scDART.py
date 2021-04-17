import sys, os
sys.path.append('../')
sys.path.insert(1, '/scDART/')

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import load_npz

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from umap import UMAP

import scDART.diffusion_dist as diff
import scDART.dataset as dataset
import scDART.model as model
import scDART.loss as loss
import scDART.train as train
import scDART.utils as utils
import scDART.post_align as palign
import scDART.benchmark as bmk
import scDART.TI as ti
import networkx as nx
import seaborn as sns

class scDART(object):

    def __init__(self, n_epochs = 700, batch_size = None, learning_rate = 5e-4, \
        gact_layers = [512, 256], proj_layers = [128, 8], ts = [20,30,50], use_anchor = False, n_anchor = None, use_potential = False):
        """\
        Description:
        ------------
            Init model
        Parameters:
        ------------
            n_epochs: number of epochs. Default: 700
            batch_size: batch size for each iteration. Default: None, divide the data into 5 batches.
            learning_rate: learning_rate parameter of sgd. Default: 5e-4
            latent_dim: latent dimensions of the model. Default 8
            ts: t used for diffusion distance calculation. Default [20,30,50]
            use_anchor: using anchor information for embedding match, default False
            n_anchor: number of anchor cells used for distance calculation, default None (exact mode)
            use_potential: use potential distance or not, default False.
            k: neighborhood size for post processing, default 3.
            l_dist_type: 'kl' or 'mse'.
        
        Return:
        ------------
            model
        """
        # TODO: fix gac_layers within, using three layers model in EMBED_CONFIG, user only need to give the latent dimensions (last layer of proj_layers).
        # include the regularization values.
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gact_layers = gact_layers
        self.proj_layers = proj_layers
        self.use_anchor = use_anchor
        self.ts = ts
        self.n_anchor = n_anchor
        
        # self.rna_dataset = None
        # self.atac_dataset = None
        # self.model_dict = None
        self.z_rna = None
        self.z_atac = None

    def fit(self, dataset):
        # TODO: include fit function
        pass
    
    def transform(self, dataset):
        # TODO: include transform function
        pass

    def load_dict(self, file = None):
        # TODO: load model dict
        pass

    def save_dict(self, file = None):
        # TODO: save model dict

    def fit_transform(self, dataset):
        #TODO: create pytorch dataset
        rna_dataset = []
        atac_dataset = []
        coarse_reg = []
        
        #TODO: Decide batchsize, libsize? if batchsize is None, then calculate it this way
        batch_size = int(max([len(rna_dataset),len(atac_dataset)])/4) if self.batch_size is None else self.batch_size
        # we don't need libsize
        # libsize = rna_dataset.get_libsize()

        train_rna_loader = DataLoader(self.rna_dataset, batch_size = batch_size, shuffle = True)
        train_atac_loader = DataLoader(self.atac_dataset, batch_size = batch_size, shuffle = True)
        test_rna_loader = DataLoader(self.rna_dataset, batch_size = len(self.rna_dataset), shuffle = False)
        test_atac_loader = DataLoader(self.atac_dataset, batch_size = len(self.atac_dataset), shuffle = False)

        #TODO: check: self.gact_layers doesn't contain first and last layer. proj_layers doesn't contain first layer
        EMBED_CONFIG = {
            # 'gact_layers': [atac_dataset.counts.shape[1], 512, 256, rna_dataset.counts.shape[1]], 
            # 'proj_layers': [rna_dataset.counts.shape[1], 128, 8], # number of nodes in each 
            'gact_layers': [atac_dataset.counts.shape[1]] + self.gact_layers + [rna_dataset.counts.shape[1]], 
            'proj_layers': [rna_dataset.counts.shape[1]] + self.proj_layers, # number of nodes in each 
            'learning_rate': self.learning_rate
        }

        self.model_dict, self.z_rna, self.z_atac = scDART_train(EMBED_CONFIG = EMBED_CONFIG, reg_mtx = coarse_reg, 
                                                        train_rna_loader = train_rna_loader, 
                                                        train_atac_loader = train_atac_loader, 
                                                        test_rna_loader = test_rna_loader, 
                                                        test_atac_loader = test_atac_loader, 
                                                        n_epochs = self.n_epochs + 1, use_anchor = self.use_anchor
                                                        )

    def scDART_train(self, EMBED_CONFIG, reg_mtx, train_rna_loader, train_atac_loader, test_rna_loader, test_atac_loader, n_epochs = 1001, use_anchor = True):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(device)

        #TODO: check parameters, fixed or hyperparamters
        # calculate the distance

        # n_anchor here is the number of anchor nodes used for distance calculation, if n_anchor is none, then use exact way of calculating the distance
        # if n_anchor is given, then use fast way
        if self.n_anchor == None:
            method = "exact"
            print("Number of anchor cells not specified, using exact mode for distance calculation instead.")
        else:
            method = "fast"
            print("Using fast mode for distance calculation. Number of anchor cells:.{:d}".format(self.n_anchor))
        
        for data in test_rna_loader:
            dist_rna = diff.diffu_distance(data["count"].numpy(), ts = self.ts, 
            use_potential = False, dr = "pca", method = method , n_anchor = self.n_anchor)

        for data in test_atac_loader:
            dist_atac = diff.diffu_distance(data["count"].numpy(), ts = self.ts, 
            use_potential = False, dr = "lsi", method = "exact", n_anchor = self.n_anchor)

        dist_rna = dist_rna/np.linalg.norm(dist_rna)
        dist_atac = dist_atac/np.linalg.norm(dist_atac)
        dist_rna = torch.FloatTensor(dist_rna).to(device)
        dist_atac = torch.FloatTensor(dist_atac).to(device)

        #TODO: dropoutrate, slope as hyperparameter or fixed?. fix them.
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

        print(model_dict)

        train.match_latent(model = model_dict, opts = opt_dict, dist_atac = dist_atac, dist_rna = dist_rna, 
                            data_loader_rna = train_rna_loader, data_loader_atac = train_atac_loader, n_epochs = n_epochs, 
                            reg_mtx = reg_mtx, reg_d = 1, reg_g = 1, reg_mmd = 1, use_anchor = use_anchor, norm = "l1", 
                            mode = "kl")

        with torch.no_grad():
            for data in test_rna_loader:
                z_rna = model_dict["encoder"](data['count'].to(device)).cpu().detach()

            for data in test_atac_loader:
                z_atac = model_dict["encoder"](model_dict["gene_act"](data['count'].to(device))).cpu().detach()

        #TODO: k hyperparameter or fixed? No put it into the init
        # post-maching
        z_rna, z_atac = palign.match_alignment(z_rna = z_rna, z_atac = z_atac, k = 10)
        z_atac, z_rna = palign.match_alignment(z_rna = z_atac, z_atac = z_rna, k = 10)

    
        return model_dict, z_rna, z_atac

    def visualize(self, mode, save_path=None, model=None):
        model = self.model_dict if model is None else model

        if mode == "embedding":
            pca_op = PCA(n_components = 2)
            z = pca_op.fit_transform(np.concatenate((z_rna.numpy(), z_atac.numpy()), axis = 0))
            z_rna_pca = z[:z_rna.shape[0],:]
            z_atac_pca = z[z_rna.shape[0]:,:]

            utils.plot_latent(z1 = z_rna_pca, z2 = z_atac_pca, anno1 = rna_dataset.cell_labels, 
                anno2 = atac_dataset.cell_labels, mode = "joint",
                save = save_path, figsize = (10,7), axis_label = "PCA")


            utils.plot_latent(z1 = z_rna_pca, z2 = z_atac_pca, anno1 = rna_dataset.cell_labels,
                anno2 = atac_dataset.cell_labels, mode = "modality", 
                save = save_path, figsize = (10,7), axis_label = "PCA")

        #TODO: TI visualization, backbone and pseudotime