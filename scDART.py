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

import script.diffusion_dist as diff
import script.dataset as dataset
import script.model as model
import script.loss as loss
import script.train as train
import script.utils as utils
import script.post_align as palign
import script.benchmark as bmk
import script.TI as ti
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

    
    def fit(self, data):

        """\
            Build scDART model using the dataset
        """

        # TODO: check: fit function is very similar to fit_transform except it does not store z_rna and z_atac
        # TODO: test this
        # TODO: Fix data preprocessing in dataset.py
        rna_count, rna_anno, rna_anchor, atac_count, atac_anno, atac_anchor, reg = data 
        self.rna_dataset = dataset.dataset(rna_count, rna_anno, rna_anchor)
        self.atac_dataset = dataset.dataset(atac_count, atac_anno, atac_anchor)
        # TODO: Fix data preprocessing
        coarse_reg = torch.FloatTensor(pd.read_csv(reg, sep = "\t", index_col = 0, header = None).values).to(self.device)

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

        self.model_dict, _, _ = train.scDART_train(EMBED_CONFIG = EMBED_CONFIG, reg_mtx = coarse_reg, 
                                                        train_rna_loader = train_rna_loader, 
                                                        train_atac_loader = train_atac_loader, 
                                                        test_rna_loader = test_rna_loader, 
                                                        test_atac_loader = test_atac_loader, 
                                                        n_epochs = self.n_epochs + 1, use_anchor = self.use_anchor,
                                                        n_anchor = self.n_anchor, ts = self.ts, reg_d = self.reg_d,
                                                        reg_g = self.reg_g, reg_mmd = self.reg_mmd, 
                                                        l_dist_type= self.l_dist_type, k = self.k, device = self.device
                                                        )
        
        print("Fit finished")
    
    def transform(self, data):
        """\
            Merge latent space
        """

        #TODO: test this
        if model_dict is None:
            print("Model does not exist. Please train a model with fit function")
            return

        # TODO: Fix data preprocessing
        rna_count, rna_anno, rna_anchor, atac_count, atac_anno, atac_anchor, reg = data 
        self.rna_dataset = dataset.dataset(rna_count, rna_anno, rna_anchor)
        self.atac_dataset = dataset.dataset(atac_count, atac_anno, atac_anchor)

        test_rna_loader = DataLoader(self.rna_dataset, batch_size = len(self.rna_dataset), shuffle = False)
        test_atac_loader = DataLoader(self.atac_dataset, batch_size = len(self.atac_dataset), shuffle = False)

        with torch.no_grad():
            for data in test_rna_loader:
                z_rna = model_dict["encoder"](data['count'].to(self.device)).cpu().detach()

            for data in test_atac_loader:
                z_atac = model_dict["encoder"](model_dict["gene_act"](data['count'].to(self.device))).cpu().detach()

        # post-maching
        self.z_rna, self.z_atac = palign.match_alignment(z_rna = z_rna, z_atac = z_atac, k = self.k)
        self.z_atac, self.z_rna = palign.match_alignment(z_rna = z_atac, z_atac = z_rna, k = self.k)

        print("Transform finished")


    def fit_transform(self, data):

        """\
            Build scDART model using the dataset. Merge latent space.
        """

        #TODO: test this
        # TODO: Fix data preprocessing
        rna_count, rna_anno, rna_anchor, atac_count, atac_anno, atac_anchor, reg = data 
        self.rna_dataset = dataset.dataset(rna_count, rna_anno, rna_anchor)
        self.atac_dataset = dataset.dataset(atac_count, atac_anno, atac_anchor)
        # TODO: Fix data preprocessing
        coarse_reg = torch.FloatTensor(pd.read_csv(reg, sep = "\t", index_col = 0, header = None).values).to(self.device)
        
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

        self.model_dict, _, _ = train.scDART_train(EMBED_CONFIG = EMBED_CONFIG, reg_mtx = coarse_reg, 
                                                        train_rna_loader = train_rna_loader, 
                                                        train_atac_loader = train_atac_loader, 
                                                        test_rna_loader = test_rna_loader, 
                                                        test_atac_loader = test_atac_loader, 
                                                        n_epochs = self.n_epochs + 1, use_anchor = self.use_anchor,
                                                        n_anchor = self.n_anchor, ts = self.ts, reg_d = self.reg_d,
                                                        reg_g = self.reg_g, reg_mmd = self.reg_mmd, 
                                                        l_dist_type= self.l_dist_type, k = self.k, device = self.device
                                                        )
        print("Fit and transform finished")



    def load_model(self, file = None):

        """\
            Load model
        """

        #TODO: check this code
        self.model_dict = torch.load(file)
        print(self.model_dict)
        print("Model loaded")


    def save_model(self, file = None):

        """\
            Save model
        """

        #TODO: check this code
        if self.model_dict is None:
            print("No model to save.")
            return
        torch.save(model, file)
        print("Model saved")

    def visualize(self, mode, save_path=None):

        """\
            Visualize merged latent space
        """

        #TODO: test this
        if self.z_rna is None or self.z_atac is None:
            print("Latent space does not exist")
            return

        if mode == "embedding":
            pca_op = PCA(n_components = 2)
            z = pca_op.fit_transform(np.concatenate((self.z_rna.numpy(), self.z_atac.numpy()), axis = 0))
            z_rna_pca = z[:z_rna.shape[0],:]
            z_atac_pca = z[z_rna.shape[0]:,:]

            utils.plot_latent(z1 = z_rna_pca, z2 = z_atac_pca, anno1 = self.rna_dataset.cell_labels, 
                anno2 = self.atac_dataset.cell_labels, mode = "joint",
                save = save_path, figsize = (10,7), axis_label = "PCA")


            utils.plot_latent(z1 = z_rna_pca, z2 = z_atac_pca, anno1 = self.rna_dataset.cell_labels,
                anno2 = self.atac_dataset.cell_labels, mode = "modality", 
                save = save_path, figsize = (10,7), axis_label = "PCA")

        if mode == "backbone" or mode == "pseudotime":
            dpt_mtx = ti.dpt(np.concatenate((z_rna, z_atac), axis = 0), n_neigh = 10)
            root_cell = 0
            pt_infer = dpt_mtx[root_cell, :]
            pt_infer[pt_infer.argsort()] = np.arange(len(pt_infer))
            pt_infer = pt_infer/np.max(pt_infer)

            groups, mean_cluster, T = ti.backbone_inf(z_rna, z_atac, resolution = 0.5)

            pca_op = PCA(n_components=2)
            ae_coord = pca_op.fit_transform(np.concatenate((z_rna, z_atac), axis = 0))
            mean_cluster = pca_op.transform(np.array(mean_cluster))

            ae_rna = ae_coord[:z_rna.shape[0],:]
            ae_atac = ae_coord[z_rna.shape[0]:,:]

            if mode == "backbone":
                utils.plot_backbone(ae_rna, ae_atac, mode = "joint", mean_cluster = mean_cluster, groups = groups, T = T, 
                    figsize=(10,7), save = save_path, anno1 = self.rna_dataset.cell_labels, anno2 = self.atac_dataset.cell_labels, axis_label = "PCA")
            if mode == "pseudotime":
                utils.plot_latent_pt(ae_rna, ae_atac, pt1 = pt_infer[:ae_rna.shape[0]], pt2 = pt_infer[ae_rna.shape[0]:], 
                    mode = "joint", save = save_path, figsize = (10,7), axis_label = "PCA")

        else:
            print("Please use embedding, backbone, or pseudotime mode")