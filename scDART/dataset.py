import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd 


class dataset(Dataset):
    """\
        Description:
        ------------
            Create Pytorch Dataset

        Parameters:
        ------------
            counts: gene count. Type: numpy ndarrary
            anchor: anchor index. Type: numpy ndarray.
        
        Return:
        ------------
            Dataset
        """
    def __init__(self, counts, anchor = None):

        assert not len(counts) == 0, "Count is empty"
        self.counts = torch.FloatTensor(counts)

        self.is_anchor = np.zeros(self.counts.shape[0]).astype("bool")    
        if anchor is not None:
            self.is_anchor[anchor] = True
        
        self.is_anchor = torch.tensor(self.is_anchor)
                   
    def __len__(self):
        return self.counts.shape[0]
    
    def __getitem__(self, idx):
        # data original data, index the index of cell, label, corresponding labels, batch, corresponding batch number
        sample = {"count": self.counts[idx,:], "index": idx, "is_anchor": self.is_anchor[idx]}
        return sample
