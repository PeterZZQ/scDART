import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd 
from scipy.sparse import load_npz

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class symsim2_rna(Dataset):
    def __init__(self, counts_dir = "./data/symsim2/rand1/GxC.txt", anno_dir = "./data/symsim2/rand1/cell_label1.txt", anchor = None, libsize = None):
        counts = pd.read_csv(counts_dir, sep = "\t", header = None).values.T
        cell_labels = pd.read_csv(anno_dir, sep = "\t")["pop"].values

        
        if counts_dir.split("/")[-2].split("_")[0] == "linear":
            idx = np.random.choice(counts.shape[0], size = 1000, replace = False)
        else:
            idx = np.arange(counts.shape[0])

        if libsize is None:
            self.libsize = np.median(np.sum(counts, axis = 1))
        else:
            self.libsize = libsize
        
        counts = counts/np.sum(counts, axis = 1)[:, None] * self.libsize 
        # minor difference after log
        counts = np.log1p(counts)

        # update the libsize after the log transformation
        self.libsize = np.mean(np.sum(counts, axis = 1))

        self.counts = torch.FloatTensor(counts[idx,:])
        self.cell_labels = cell_labels[idx]

        if anchor is not None:
            self.is_anchor = (self.cell_labels == anchor)
        else:
            self.is_anchor = np.zeros(self.cell_labels.shape[0]).astype("bool")
        
        self.is_anchor = torch.tensor(self.is_anchor)
        self.use_clust = False

    def __len__(self):
        return self.counts.shape[0]
    
    def get_libsize(self):
        return self.libsize
    
    def __getitem__(self, idx):
        # data original data, index the index of cell, label, corresponding labels, batch, corresponding batch number
        if self.use_clust:
            sample = {"count": self.counts[idx,:], "index": idx, "is_anchor": self.is_anchor[idx], "groups": self.groups[idx]}
        else:
            sample = {"count": self.counts[idx,:], "index": idx, "is_anchor": self.is_anchor[idx]}
        return sample

class symsim2_atac(Dataset):
    def __init__(self, counts_dir = "./data/symsim2/rand2/RxC.txt", anno_dir = "./data/symsim2/rand2/cell_label2.txt", anchor = None):
        counts = pd.read_csv(counts_dir, sep = "\t", header = None).values.T
        counts = np.where(counts < 1, 0, 1)
        cell_labels = pd.read_csv(anno_dir, sep = "\t")["pop"].values
        
        if counts_dir.split("/")[-2].split("_")[0] == "linear":
            idx = np.random.choice(counts.shape[0], size = 1000, replace = False)
        else:
            idx = np.arange(counts.shape[0])
        
        self.counts = torch.FloatTensor(counts[idx,:])
        self.cell_labels = cell_labels[idx]
        if anchor is not None:
            self.is_anchor = (self.cell_labels == anchor)
        else:
            self.is_anchor = np.zeros(self.cell_labels.shape[0]).astype("bool")
        
        self.is_anchor = torch.tensor(self.is_anchor)
        self.use_clust = False

    def __len__(self):
        return self.counts.shape[0]
    
    
    def __getitem__(self, idx):
        if self.use_clust:
            sample = {"count": self.counts[idx,:], "index": idx, "is_anchor": self.is_anchor[idx], "groups": self.groups[idx]}
        else:
            sample = {"count": self.counts[idx,:], "index": idx, "is_anchor": self.is_anchor[idx]}
        return sample

class hema_rna(Dataset):
    def __init__(self, counts_dir = "./data/hema/counts_rna.csv", anno_dir = "./data/hema/anno_rna.txt", anchor = None):
        counts = pd.read_csv(counts_dir, index_col=0).values
        cell_labels = []

        with open(anno_dir, "r") as fp:
            for i in fp:
                cell_labels.append(i.strip("\n"))
        cell_labels = np.array(cell_labels)

        # calculate the libsize if needed
        self.libsize = np.mean(np.sum(counts, axis = 1))

        self.counts = torch.FloatTensor(counts)
        self.cell_labels = cell_labels

        if anchor is not None:
            self.is_anchor = (self.cell_labels == anchor)
        else:
            self.is_anchor = np.zeros(self.cell_labels.shape[0]).astype("bool")    
        
        self.is_anchor = torch.tensor(self.is_anchor)

    def get_libsize(self):
        return self.libsize
                   
    def __len__(self):
        return self.counts.shape[0]
    
    def __getitem__(self, idx):
        # data original data, index the index of cell, label, corresponding labels, batch, corresponding batch number
        sample = {"count": self.counts[idx,:], "index": idx, "is_anchor": self.is_anchor[idx]}
        return sample
        
class endo_rna(Dataset):
    def __init__(self, counts_dir = "./data/Endo/counts_rna.csv", anno_dir = "./data/Endo/anno_rna.txt", anchor = None):
        counts = pd.read_csv(counts_dir, index_col=0).values
        cell_labels = []
        with open(anno_dir, "r") as fp:
            for i in fp:
                cell_labels.append(i.strip("\n"))
        cell_labels = np.array(cell_labels)

        # calculate the libsize if needed
        self.libsize = np.mean(np.sum(counts, axis = 1))
        
        # get processed count matrix 
        self.counts = torch.FloatTensor(counts)
        self.cell_labels = cell_labels

        if anchor is not None:
            self.is_anchor = (self.cell_labels == anchor)
        else:
            self.is_anchor = np.zeros(self.cell_labels.shape[0]).astype("bool")    
        
        self.is_anchor = torch.tensor(self.is_anchor)

    def get_libsize(self):
        return self.libsize
        
    def __len__(self):
        return self.counts.shape[0]
    
    def __getitem__(self, idx):
        sample = {"count": self.counts[idx,:], "index": idx, "is_anchor": self.is_anchor[idx]}
        return sample

class hema_atac(Dataset):
    def __init__(self, counts_dir = "./data/hema/counts_atac.csv", anno_dir = "./data/hema/anno_atac.txt", anchor = None):

        counts = pd.read_csv(counts_dir, index_col=0).values
        counts = np.where(counts < 1, 0, 1)

        cell_labels = []
        with open(anno_dir, "r") as fp:
            for i in fp:
                cell_labels.append(i.strip("\n"))
        cell_labels = np.array(cell_labels)
        
        # get processed count matrix 
        self.counts = torch.FloatTensor(counts)
        self.cell_labels = cell_labels
        if anchor is not None:
            self.is_anchor = (self.cell_labels == anchor)
        else:
            self.is_anchor = np.zeros(self.cell_labels.shape[0]).astype("bool")    
        
        self.is_anchor = torch.tensor(self.is_anchor)

    def __len__(self):
        return self.counts.shape[0]
    
    def __getitem__(self, idx):
        sample = {"count": self.counts[idx,:], "index": idx, "is_anchor": self.is_anchor[idx]}
        return sample


class endo_atac(Dataset):
    def __init__(self, counts_dir = "./data/Endo/counts_atac.csv", anno_dir = "./data/Endo/anno_atac.txt", anchor = None):

        counts = pd.read_csv(counts_dir, index_col=0).values
        counts = np.where(counts < 1, 0, 1)
        cell_labels = []
        with open(anno_dir, "r") as fp:
            for i in fp:
                cell_labels.append(i.strip("\n"))
        cell_labels = np.array(cell_labels)

        self.counts = torch.FloatTensor(counts)
        self.cell_labels = cell_labels
        if anchor is not None:
            self.is_anchor = (self.cell_labels == anchor)
        else:
            self.is_anchor = np.zeros(self.cell_labels.shape[0]).astype("bool")
        
        self.is_anchor = torch.tensor(self.is_anchor)
        
    def __len__(self):
        return self.counts.shape[0]
    
    def __getitem__(self, idx):
        sample = {"count": self.counts[idx,:], "index": idx, "is_anchor": self.is_anchor[idx]}
        return sample


class hema_full_rna(Dataset):
    def __init__(self, counts_dir = "./data/hema_full/compress/counts_rna.npz", anno_dir = "./data/hema_full/anno_rna.csv", anchor = None):
        
        counts = load_npz(counts_dir).todense()
        cell_labels = pd.read_csv(anno_dir, index_col = 0).values.squeeze()
        idx = np.where((cell_labels != "08_GMP.Neut")&(cell_labels != "09_pDC")&(cell_labels != "10_cDC"))[0]
        
        # calculate the libsize if needed
        self.libsize = np.mean(np.sum(counts, axis = 1))
        
        # get processed count matrix 
        self.counts = torch.FloatTensor(counts[idx,:])
        self.cell_labels = cell_labels[idx]

        if anchor is not None:
            self.is_anchor = (self.cell_labels == anchor)
        else:
            self.is_anchor = np.zeros(self.cell_labels.shape[0]).astype("bool")    
        
        self.is_anchor = torch.tensor(self.is_anchor)

    def get_libsize(self):
        return self.libsize
        
    def __len__(self):
        return self.counts.shape[0]
    
    def __getitem__(self, idx):
        sample = {"count": self.counts[idx,:], "index": idx, "is_anchor": self.is_anchor[idx]}
        return sample


class hema_full_atac(Dataset):
    def __init__(self, counts_dir = "./data/hema_full/compress/counts_atac.npz", anno_dir = "./data/hema_full/anno_atac.txt", anchor = None):

        counts = load_npz(counts_dir).todense()
        counts = np.where(counts < 1, 0, 1)

        cell_labels = pd.read_csv(anno_dir, index_col = 0).values.squeeze()
        idx = np.where((cell_labels != "08_GMP.Neut")&(cell_labels != "09_pDC")&(cell_labels != "10_cDC"))[0]
        
        # get processed count matrix 
        self.counts = torch.FloatTensor(counts[idx,:])
        self.cell_labels = cell_labels[idx]
        if anchor is not None:
            self.is_anchor = (self.cell_labels == anchor)
        else:
            self.is_anchor = np.zeros(self.cell_labels.shape[0]).astype("bool")    
        
        self.is_anchor = torch.tensor(self.is_anchor)

    def __len__(self):
        return self.counts.shape[0]
    
    def __getitem__(self, idx):
        sample = {"count": self.counts[idx,:], "index": idx, "is_anchor": self.is_anchor[idx]}
        return sample

class braincortex_rna(Dataset):
    def __init__(self, counts_dir = "./data/snare-seq/counts_rna.csv", anno_dir = "./data/snare-seq/anno.txt", anchor = None):

        counts = pd.read_csv(counts_dir, index_col=0).values
        cell_labels = []
        with open(anno_dir, "r") as fp:
            for i in fp:
                cell_labels.append(i.strip("\n"))
        cell_labels = np.array(cell_labels)

        # calculate the libsize if needed
        self.libsize = np.mean(np.sum(counts, axis = 1))
        
        # get processed count matrix 
        self.counts = torch.FloatTensor(counts)
        self.cell_labels = cell_labels

        if anchor is not None:
            self.is_anchor = (self.cell_labels == anchor)
        else:
            self.is_anchor = np.zeros(self.cell_labels.shape[0]).astype("bool")    
        
        self.is_anchor = torch.tensor(self.is_anchor)

    def get_libsize(self):
        return self.libsize
        
    def __len__(self):
        return self.counts.shape[0]
    
    def __getitem__(self, idx):
        sample = {"count": self.counts[idx,:], "index": idx, "is_anchor": self.is_anchor[idx]}
        return sample

class braincortex_atac(Dataset):
    def __init__(self, counts_dir = "./data/snare-seq/counts_atac.npz", anno_dir = "./data/snare-seq/anno.txt", anchor = None):

        counts = load_npz(counts_dir).todense()
        counts = np.where(counts < 1, 0, 1)

        cell_labels = []
        with open(anno_dir, "r") as fp:
            for i in fp:
                cell_labels.append(i.strip("\n"))
        cell_labels = np.array(cell_labels)
        
        # get processed count matrix 
        self.counts = torch.FloatTensor(counts)
        self.cell_labels = cell_labels
        if anchor is not None:
            self.is_anchor = (self.cell_labels == anchor)
        else:
            self.is_anchor = np.zeros(self.cell_labels.shape[0]).astype("bool")    
        
        self.is_anchor = torch.tensor(self.is_anchor)

    def __len__(self):
        return self.counts.shape[0]
    
    def __getitem__(self, idx):
        sample = {"count": self.counts[idx,:], "index": idx, "is_anchor": self.is_anchor[idx]}
        return sample