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

    def __init__(self, n_epochs = 700, batch_size = None, learning_rate = 3e-4, latent_dim = 8, \
        ts = [30, 50, 70], use_anchor = False, use_potential = False, k = 3, \
        reg_d = 1, reg_g = 1, reg_mmd = 1, l_dist_type = 'kl', seed = 0,\
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

        """\
        Description:
        ------------
            Init model

        Parameters:
        ------------
            n_epochs: int, default = 700
                number of epochs.
            
            batch_size: int, default = None
                batch size for each iteration. If None, divide the data into 5 batches.
            
            learning_rate: float, default: 5e-4
                learning_rate parameter of sgd. 
            
            latent_dim: int, default: 8
                latent dimensions of the model.
            
            ts: list of int, default [30, 50, 70]
                t used for diffusion distance calculation.
            
            use_anchor: bool, default: False
                using anchor information for embedding match.
            
            use_potential: bool, default: False
                use potential distance or not.
            
            k: int, default: 3
                neighborhood size for post processing. Default: 3.
            
            reg_d: int or float, default: 1
                distance regularization. Default: 1

            reg_g: int or float, default: 1
                genact regularization. Default: 1

            reg_mmd: int or float, default: 1
                mmd regularization. Default: 1

            l_dist_type: default: 'kl'
                'kl' or 'mse'.

            device: torch device, default: torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                the torch device on which memory is allocated.
        
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

        self.seed = seed

        # set the random seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)


    
    def fit(self, rna_count, atac_count, reg, rna_anchor = None, atac_anchor = None):

        """\
        Description:
        ------------
            Train scDART model

        Parameters:
        ------------
            rna_count: ndarray
                rna gene count
            
            atac_count: ndarray
                atac gene count
            
            reg: ndarray
                gene activation

            rna_anchor: ndarray, default: None
                rna anchor index

            atac_anchor: ndarray, default: None
                atac anchor index
        
        Return:
        ------------
            self
        """

        self.rna_dataset = dataset.dataset(rna_count, rna_anchor)
        self.atac_dataset = dataset.dataset(atac_count, atac_anchor)
        coarse_reg = torch.FloatTensor(reg).to(self.device)

        # batch_size = int(max([len(self.rna_dataset),len(self.atac_dataset)])/4) if self.batch_size is None else self.batch_size
        batch_size = int(max([len(self.rna_dataset),len(self.atac_dataset)])/4)
        
        train_rna_loader = DataLoader(self.rna_dataset, batch_size = batch_size, shuffle = True)
        train_atac_loader = DataLoader(self.atac_dataset, batch_size = batch_size, shuffle = True)

        print("Loaded Dataset")

        EMBED_CONFIG = {
            'gact_layers': [self.atac_dataset.counts.shape[1], 1024, 512, self.rna_dataset.counts.shape[1]], 
            'proj_layers': [self.rna_dataset.counts.shape[1], 512, 128, self.latent_dim], # number of nodes in each 
        }

        # calculate the diffusion distance
        dist_rna = diff.diffu_distance(self.rna_dataset.counts.numpy(), ts = self.ts,
                                        use_potential = self.use_potential, dr = "pca", n_components = 30)

        dist_atac = diff.diffu_distance(self.atac_dataset.counts.numpy(), ts = self.ts,
                                        use_potential = self.use_potential, dr = "lsi", n_components = 30)
        
        dist_rna = dist_rna/np.linalg.norm(dist_rna)
        dist_atac = dist_atac/np.linalg.norm(dist_atac)
        dist_rna = torch.FloatTensor(dist_rna).to(self.device)
        dist_atac = torch.FloatTensor(dist_atac).to(self.device)

        # initialize the model
        gene_act = model.gene_act(features = EMBED_CONFIG["gact_layers"], dropout_rate = 0.0, negative_slope = 0.2).to(self.device)
        encoder = model.Encoder(features = EMBED_CONFIG["proj_layers"], dropout_rate = 0.0, negative_slope = 0.2).to(self.device)
        self.model_dict = {"gene_act": gene_act, "encoder": encoder}

        opt_genact = torch.optim.Adam(gene_act.parameters(), lr = self.learning_rate)
        opt_encoder = torch.optim.Adam(encoder.parameters(), lr = self.learning_rate)
        opt_dict = {"gene_act": opt_genact, "encoder": opt_encoder}

        # training models
        train.match_latent(model = self.model_dict, opts = opt_dict, dist_atac = dist_atac, dist_rna = dist_rna, 
                        data_loader_rna = train_rna_loader, data_loader_atac = train_atac_loader, n_epochs = self.n_epochs + 1, 
                        reg_mtx = coarse_reg, reg_d = self.reg_d, reg_g = self.reg_g, reg_mmd = self.reg_mmd, use_anchor = self.use_anchor, norm = "l1", 
                        mode = self.l_dist_type)


        print("Fit finished")

        return(self)
    

    def transform(self, rna_count, atac_count, rna_anchor = None, atac_anchor = None):
        
        """\
        Description:
        ------------
            Induce rna latent space and atac latent space 

        Parameters:
        ------------
            rna_count: ndarray
                rna gene count
            
            atac_count: ndarray
                atac gene count
            
            reg: ndarray
                gene activation

            rna_anchor: ndarray, default: None
                rna anchor index

            atac_anchor: ndarray, default: None
                atac anchor index
        
        Return:
        ------------
            z_rna: ndarray
                rna latent space
            z_atac: ndarray
                atac latent space
        """

        assert self.model_dict is not None, "Model does not exist. Please train a model with fit function"

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
        self.z_rna, self.z_atac = self.z_rna.numpy(), self.z_atac.numpy()

        print("Transform finished")

        return self.z_rna, self.z_atac


    def fit_transform(self, rna_count, atac_count, reg, rna_anchor = None, atac_anchor = None):

        """\
        Description:
        ------------
            Train scDART model. Induce rna latent space and atac latent space.

        Parameters:
        ------------
            rna_count: ndarray
                rna gene count
            
            atac_count: ndarray
                atac gene count
            
            reg: ndarray
                gene activation

            rna_anchor: ndarray, default: None
                rna anchor index

            atac_anchor: ndarray, default: None
                atac anchor index
        
        Return:
        ------------
            z_rna: ndarray
                rna latent space
            z_atac: ndarray
                atac latent space
        """

        assert not len(reg) == 0, "Gene activation is empty"
        self.rna_dataset = dataset.dataset(rna_count, rna_anchor)
        self.atac_dataset = dataset.dataset(atac_count, atac_anchor)
        coarse_reg = torch.FloatTensor(reg).to(self.device)

        batch_size = int(max([len(self.rna_dataset),len(self.atac_dataset)])/4) if self.batch_size is None else self.batch_size
        
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

        # calculate the diffusion distance
        dist_rna = diff.diffu_distance(self.rna_dataset.counts.numpy(), ts = self.ts,
                                        use_potential = self.use_potential, dr = "pca", n_components = 30)

        dist_atac = diff.diffu_distance(self.atac_dataset.counts.numpy(), ts = self.ts,
                                        use_potential = self.use_potential, dr = "lsi", n_components = 30)
        
        dist_rna = dist_rna/np.linalg.norm(dist_rna)
        dist_atac = dist_atac/np.linalg.norm(dist_atac)
        dist_rna = torch.FloatTensor(dist_rna).to(self.device)
        dist_atac = torch.FloatTensor(dist_atac).to(self.device)

        # initialize the model
        gene_act = model.gene_act(features = EMBED_CONFIG["gact_layers"], dropout_rate = 0.0, negative_slope = 0.2).to(self.device)
        encoder = model.Encoder(features = EMBED_CONFIG["proj_layers"], dropout_rate = 0.0, negative_slope = 0.2).to(self.device)
        self.model_dict = {"gene_act": gene_act, "encoder": encoder}

        opt_genact = torch.optim.Adam(gene_act.parameters(), lr = self.learning_rate)
        opt_encoder = torch.optim.Adam(encoder.parameters(), lr = self.learning_rate)
        opt_dict = {"gene_act": opt_genact, "encoder": opt_encoder}

        # training models
        train.match_latent(model = self.model_dict, opts = opt_dict, dist_atac = dist_atac, dist_rna = dist_rna, 
                        data_loader_rna = train_rna_loader, data_loader_atac = train_atac_loader, n_epochs = self.n_epochs, 
                        reg_mtx = coarse_reg, reg_d = self.reg_d, reg_g = self.reg_g, reg_mmd = self.reg_mmd, use_anchor = self.use_anchor, norm = "l1", 
                        mode = self.l_dist_type)
                        
        with torch.no_grad():
            for data in test_rna_loader:
                self.z_rna = self.model_dict["encoder"](data['count'].to(self.device)).cpu().detach()

            for data in test_atac_loader:
                self.z_atac = self.model_dict["encoder"](self.model_dict["gene_act"](data['count'].to(self.device))).cpu().detach()

        # post-maching
        self.z_rna, self.z_atac = palign.match_alignment(z_rna = self.z_rna, z_atac = self.z_atac, k = self.k)
        self.z_atac, self.z_rna = palign.match_alignment(z_rna = self.z_atac, z_atac = self.z_rna, k = self.k)
        self.z_rna, self.z_atac = self.z_rna.numpy(), self.z_atac.numpy()

        print("Fit and transform finished")

        return self.z_rna, self.z_atac



    def load_model(self, save_path):

        """\
        Description:
        ------------
            Load model

        Parameters:
        ------------
            save_path: str
                path to file
        
        Return:
        ------------
            None
        """

        self.model_dict = torch.load(save_path)
        print(self.model_dict)
        print("Model loaded")


    def save_model(self, save_path):

        """\
        Description:
        ------------
            Init model

        Parameters:
        ------------
            n_epochs: 
                number of epochs. Default: 700
        
        Return:
        ------------
            None
        """

        assert self.model_dict is not None, "No model to save."

        torch.save(self.model_dict, save_path)
        print("Model saved")


    def visualize(self, rna_anno = None, atac_anno = None, mode = "embedding", save_path = None, **kwargs):

        """\
        Description:
        ------------
            visualize joint latent space

        Parameters:
        ------------
            rna_anno: ndarray, default: None
                rna cell label
            
            atac_anno: ndarray, default: None
                atac cell label

            mode: "embedding", "backbone", "pseudotime"
                visulization mode

            save_path: str, default: None
                path to save the visualization

            resolution: float, default: 0.5
                parameter of ti backbone

            n_neigh: int, default: 10
                parameter of ti backbone

            fig_size: int or list of int, default: (10, 7)
                size of the figure
        
        Return:
        ------------
            None
        """

        _kwargs = {
            "resolution": 0.5,
            "n_neigh": 10,
            "fig_size": (10, 7)
        }
        _kwargs.update(kwargs)

        assert self.z_rna is not None and self.z_atac is not None, "Latent space does not exist"

        if mode == "embedding":
            pca_op = PCA(n_components = 2)
            z = pca_op.fit_transform(np.concatenate((self.z_rna, self.z_atac), axis = 0))
            z_rna_pca = z[:self.z_rna.shape[0],:]
            z_atac_pca = z[self.z_rna.shape[0]:,:]

            if rna_anno is not None and atac_anno is not None:
                utils.plot_latent(z1 = z_rna_pca, z2 = z_atac_pca, anno1 = rna_anno, 
                    anno2 = atac_anno, mode = "joint",
                    save = save_path, figsize = _kwargs['fig_size'], axis_label = "PCA")


            utils.plot_latent(z1 = z_rna_pca, z2 = z_atac_pca, anno1 = rna_anno,
                anno2 = atac_anno, mode = "modality", 
                save = save_path, figsize = _kwargs['fig_size'], axis_label = "PCA")

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
                    figsize=_kwargs['fig_size'], save = save_path, anno1 = rna_anno, anno2 = atac_anno, axis_label = "PCA")
            if mode == "pseudotime":
                utils.plot_latent_pt(ae_rna, ae_atac, pt1 = pt_infer[:ae_rna.shape[0]], pt2 = pt_infer[ae_rna.shape[0]:], 
                    mode = "joint", save = save_path, figsize = _kwargs['fig_size'], axis_label = "PCA")

        else:
            print("Please use embedding, backbone, or pseudotime mode")