"""Utility classes and methods.
"""

import numpy as np
import torch
import torch.utils.data as data
import ujson as json

class TextDataset(data.Dataset):
    """Text dataset used to train a language model

    Each item in the dataset is a tuple (x,y) where:
        x : sequence of input words
        y : sequence of target output words
    
    Args:
        data_path (str): Path to .npz file containeing pre-processed dataset.
        sequence_length (int): Length of input and target sequences
    """

    def __init__(self, data_path, sequence_length):
        super(TextDataset, self).__init__()
        
        dataset = np.load(data_path)
        self.text_idxs = torch.from_numpy(dataset['text_idxs']).long()
        self.sequence_length = sequence_length
    
    def __getitem__(self, idx):
        idx = idx * self.sequence_length
        return (self.text_idxs[idx:idx+self.sequence_length], self.text_idxs[idx+1:idx+self.sequence_length+1])
    
    def __len__(self):
        return len(self.text_idxs)//self.sequence_length


def torch_from_json(path, dtype=torch.float32):
    """Load a PyTorch Tensor from a JSON file.
    Taken from:
        https://github.com/michiyasunaga/squad/blob/main/util.py
    
    Args:
        path (str): Path to the JSON file to load.
        dtype (torch.dtype): Data type of loaded array.
    
    Returns:
        tensor (torch.Tensor): Tensor loaded from JSON file.    
    """
    
    with open(path, 'r') as fh:
        array = np.array(json.load(fh))
    
    tensor = torch.from_numpy(array).type(dtype)
    
    return tensor