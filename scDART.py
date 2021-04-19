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

import src.diffusion_dist as diff
import src.dataset as dataset
import src.model as model
import src.loss as loss
import src.train as train
import src.utils as utils
import src.post_align as palign
import src.benchmark as bmk
import src.TI as ti
import networkx as nx
import seaborn as sns

class scDART(object):

    def __init__(self, n_epochs = 700, batch_size = None, learning_rate = 5e-4, latent_dim = 8, \
        ts = [20,30,50], use_anchor = False, n_anchor = None, use_potential = False, k = 3, \
        reg_d = 1, reg_g = 1, reg_mmd = 1, l_dist_type = 'kl', 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

        """\
        Description:
        ------------
            Init model

        Parameters:
        ------------
            n_epochs: number of epochs. Default: 700
            batch_size: batch size for each iteration. Default: None, divide the data into 5 batches.
            learning_rate: learning_rate parameter of sgd. Default: 5e-4
            latent_dim: latent dimensions of the model. Default: 8
            ts: t used for diffusion distance calculation. Default [20,30,50]
            use_anchor: using anchor information for embedding match. Default: False
            n_anchor: number of anchor cells used for distance calculation. Default: None (exact mode)
            use_potential: use potential distance or not. Default: False.
            k: neighborhood size for post processing. Default: 3.
            reg_d: distance regularization. Default: 1
            reg_g: genact regularization. Default: 1
            reg_mmd: mmd regularization. Default: 1
            l_dist_type: 'kl' or 'mse'.
            device: the torch device on which memory is allocated.
        
        Return:
        ------------
            model
        """

        # TODO: include the regularization values.
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.latent_dim = latent_dim
        self.use_anchor = use_anchor
        self.ts = ts
        self.n_anchor = n_anchor
        self.use_potential = use_potential
        self.k = k
        self.reg_d = reg_d
        self.reg_g = reg_g
        self.reg_mmd = reg_mmd
        self.l_dist_type = l_dist_type
        self.device = device
        
        self.rna_dataset = None
        self.atac_dataset = None
        self.model_dict = None
        self.z_rna = None
        self.z_atac = None

    
    def fit(self, rna_count, atac_count, reg, rna_anchor = None, atac_anchor = None):

        """\
            Build scDART model using the dataset
        """

        # TODO: check: fit function is very similar to fit_transform except it does not store z_rna and z_atac
        # TODO: test this
        # TODO: Fix data preprocessing in dataset.py
        self.rna_dataset = dataset.dataset(rna_count, rna_anchor)
        self.atac_dataset = dataset.dataset(atac_count, atac_anchor)
        coarse_reg = torch.FloatTensor(reg).to(self.device)

        batch_size = int(max([len(self.rna_dataset),len(self.atac_dataset)])/5) if self.batch_size is None else self.batch_size
        
        train_rna_loader = DataLoader(self.rna_dataset, batch_size = batch_size, shuffle = True)
        train_atac_loader = DataLoader(self.atac_dataset, batch_size = batch_size, shuffle = True)
        test_rna_loader = DataLoader(self.rna_dataset, batch_size = len(self.rna_dataset), shuffle = False)
        test_atac_loader = DataLoader(self.atac_dataset, batch_size = len(self.atac_dataset), shuffle = False)

        print("Loaded Dataset")

        EMBED_CONFIG = {
            'gact_layers': [self.atac_dataset.counts.shape[1], 512, 256, self.rna_dataset.counts.shape[1]], 
            'proj_layers': [self.rna_dataset.counts.shape[1], 128] + [self.latent_dim], # number of nodes in each 
            'learning_rate': self.learning_rate
        }

        self.model_dict = train.scDART_train(EMBED_CONFIG = EMBED_CONFIG, reg_mtx = coarse_reg, 
                                                        train_rna_loader = train_rna_loader, 
                                                        train_atac_loader = train_atac_loader, 
                                                        test_rna_loader = test_rna_loader, 
                                                        test_atac_loader = test_atac_loader, 
                                                        n_epochs = self.n_epochs + 1, use_anchor = self.use_anchor,
                                                        n_anchor = self.n_anchor, ts = self.ts, reg_d = self.reg_d,
                                                        reg_g = self.reg_g, reg_mmd = self.reg_mmd, 
                                                        l_dist_type= self.l_dist_type, device = self.device
                                                        )
        
        print("Fit finished")
    
    def transform(self, rna_count, atac_count, rna_anchor = None, atac_anchor = None):
        """\
            Merge latent space
        """

        #TODO: test this
        if self.model_dict is None:
            assert("Model does not exist. Please train a model with fit function")

        # TODO: Fix data preprocessing
        self.rna_dataset = dataset.dataset(rna_count, rna_anchor)
        self.atac_dataset = dataset.dataset(atac_count, atac_anchor)

        test_rna_loader = DataLoader(self.rna_dataset, batch_size = len(self.rna_dataset), shuffle = False)
        test_atac_loader = DataLoader(self.atac_dataset, batch_size = len(self.atac_dataset), shuffle = False)

        with torch.no_grad():
            for data in test_rna_loader:
                self.z_rna = self.model_dict["encoder"](data['count'].to(self.device)).cpu().detach()

            for data in test_atac_loader:
                self.z_atac = self.model_dict["encoder"](self.model_dict["gene_act"](data['count'].to(self.device))).cpu().detach()

        # post-maching
        self.z_rna, self.z_atac = palign.match_alignment(z_rna = self.z_rna, z_atac = self.z_atac, k = self.k)
        self.z_atac, self.z_rna = palign.match_alignment(z_rna = self.z_atac, z_atac = self.z_rna, k = self.k)

        print("Transform finished")

        return self.z_rna, self.z_atac


    def fit_transform(self, rna_count, atac_count, reg, rna_anchor = None, atac_anchor = None):

        """\
            Build scDART model using the dataset. Merge latent space.
        """

        #TODO: test this
        self.rna_dataset = dataset.dataset(rna_count, rna_anchor)
        self.atac_dataset = dataset.dataset(atac_count, atac_anchor)
        if len(reg) == 0:
            assert("Gene activation is empty")
        coarse_reg = torch.FloatTensor(reg).to(self.device)
        
        batch_size = int(max([len(self.rna_dataset),len(self.atac_dataset)])/5) if self.batch_size is None else self.batch_size

        train_rna_loader = DataLoader(self.rna_dataset, batch_size = batch_size, shuffle = True)
        train_atac_loader = DataLoader(self.atac_dataset, batch_size = batch_size, shuffle = True)
        test_rna_loader = DataLoader(self.rna_dataset, batch_size = len(self.rna_dataset), shuffle = False)
        test_atac_loader = DataLoader(self.atac_dataset, batch_size = len(self.atac_dataset), shuffle = False)

        print("Loaded dataset")

        #TODO: check layers
        EMBED_CONFIG = {
            'gact_layers': [self.atac_dataset.counts.shape[1], 512, 256, self.rna_dataset.counts.shape[1]], 
            'proj_layers': [self.rna_dataset.counts.shape[1], 128] + [self.latent_dim], # number of nodes in each 
            'learning_rate': self.learning_rate
        }

        self.model_dict = train.scDART_train(EMBED_CONFIG = EMBED_CONFIG, reg_mtx = coarse_reg, 
                                                        train_rna_loader = train_rna_loader, 
                                                        train_atac_loader = train_atac_loader, 
                                                        test_rna_loader = test_rna_loader, 
                                                        test_atac_loader = test_atac_loader, 
                                                        n_epochs = self.n_epochs + 1, use_anchor = self.use_anchor,
                                                        n_anchor = self.n_anchor, ts = self.ts, reg_d = self.reg_d,
                                                        reg_g = self.reg_g, reg_mmd = self.reg_mmd, 
                                                        l_dist_type= self.l_dist_type, device = self.device
                                                        )
        
        with torch.no_grad():
            for data in test_rna_loader:
                self.z_rna = self.model_dict["encoder"](data['count'].to(self.device)).cpu().detach()

            for data in test_atac_loader:
                self.z_atac = self.model_dict["encoder"](self.model_dict["gene_act"](data['count'].to(self.device))).cpu().detach()

        # post-maching
        self.z_rna, self.z_atac = palign.match_alignment(z_rna = self.z_rna, z_atac = self.z_atac, k = self.k)
        self.z_atac, self.z_rna = palign.match_alignment(z_rna = self.z_atac, z_atac = self.z_rna, k = self.k)
        print("Fit and transform finished")

        return self.z_rna, self.z_atac



    def load_model(self, save_path = None):

        """\
            Load model
        """

        #TODO: check this code
        self.model_dict = torch.load(save_path)
        print(self.model_dict)
        print("Model loaded")


    def save_model(self, save_path = None):

        """\
            Save model
        """

        #TODO: check this code
        if self.model_dict is None:
            assert("No model to save.")

        torch.save(self.model_dict, save_path)
        print("Model saved")

    def visualize(self, rna_anno = None, atac_anno = None, mode = "embedding", save_path = None, **kwargs):

        """\
            Visualize merged latent space
        """
        _kwargs = {
            "resolution": 0.5,
            "n_neigh": 10,
            "fig_size": (10, 7)
        }
        _kwargs.update(kwargs)

        #TODO: test this
        if self.z_rna is None or self.z_atac is None:
            assert("Latent space does not exist")

        if mode == "embedding":
            pca_op = PCA(n_components = 2)
            z = pca_op.fit_transform(np.concatenate((self.z_rna.numpy(), self.z_atac.numpy()), axis = 0))
            z_rna_pca = z[:self.z_rna.shape[0],:]
            z_atac_pca = z[self.z_rna.shape[0]:,:]

            if rna_anno is not None and atac_anno is not None:
                utils.plot_latent(z1 = z_rna_pca, z2 = z_atac_pca, anno1 = rna_anno, 
                    anno2 = atac_anno, mode = "joint",
                    save = save_path, figsize = _kwargs['figsize'], axis_label = "PCA")


            utils.plot_latent(z1 = z_rna_pca, z2 = z_atac_pca, anno1 = rna_anno,
                anno2 = atac_anno, mode = "modality", 
                save = save_path, figsize = _kwargs['figsize'], axis_label = "PCA")

        if mode == "backbone" or mode == "pseudotime":
            dpt_mtx = ti.dpt(np.concatenate((self.z_rna, self.z_atac), axis = 0), n_neigh = _kwargs['n_neigh'])
            root_cell = 0
            pt_infer = dpt_mtx[root_cell, :]
            pt_infer[pt_infer.argsort()] = np.arange(len(pt_infer))
            pt_infer = pt_infer/np.max(pt_infer)

            groups, mean_cluster, T = ti.backbone_inf(self.z_rna, self.z_atac, resolution = _kwargs['resolution'])

            pca_op = PCA(n_components=2)
            ae_coord = pca_op.fit_transform(np.concatenate((self.z_rna, self.z_atac), axis = 0))
            mean_cluster = pca_op.transform(np.array(mean_cluster))

            ae_rna = ae_coord[:self.z_rna.shape[0],:]
            ae_atac = ae_coord[self.z_rna.shape[0]:,:]

            if mode == "backbone":
                utils.plot_backbone(ae_rna, ae_atac, mode = "joint", mean_cluster = mean_cluster, groups = groups, T = T, 
                    figsize=_kwargs['figsize'], save = save_path, anno1 = rna_anno, anno2 = atac_anno, axis_label = "PCA")
            if mode == "pseudotime":
                utils.plot_latent_pt(ae_rna, ae_atac, pt1 = pt_infer[:ae_rna.shape[0]], pt2 = pt_infer[ae_rna.shape[0]:], 
                    mode = "joint", save = save_path, figsize = _kwargs['figsize'], axis_label = "PCA")

        else:
            assert("Please use embedding, backbone, or pseudotime mode")