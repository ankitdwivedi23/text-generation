"""Utility classes and methods.
Reference:
    https://github.com/michiyasunaga/squad/blob/main/util.py
"""

import numpy as np
import queue
import os
import shutil
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


class CheckpointSaver:
    """Class to save and load model checkpoints.

    Save the best checkpoints as measured by a metric value passed into the
    `save` method. Overwrite checkpoints with better checkpoints once
    `max_checkpoints` have been saved.

    Args:
        save_dir (str): Directory to save checkpoints.
        max_checkpoints (int): Maximum number of checkpoints to keep before overwriting old ones.
        metric_name (str): Name of metric used to determine best model.
        maximize_metric (bool): If true, best checkpoint is that which maximizes the metric value
                                passed in via `save`. Otherwise, best checkpoint minimizes the metric.
    """
    def __init__(self, save_dir, max_checkpoints, metric_name,
                maximize_metric=False):
        super(CheckpointSaver, self).__init__()

        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.maximize_metric = maximize_metric
        self.best_val = None
        self.ckpt_paths = queue.PriorityQueue()
        print(f"Saver will {'max' if maximize_metric else 'min'}imize {metric_name}...")

    def is_best(self, metric_val):
        """Check whether `metric_val` is the best seen so far.
        
        Args:
            metric_val (float): Metric value to compare to prior checkpoints.        
        """
        if metric_val is None:
            # No metric reported
            return False
        
        if self.best_val is None:
            # No checkpoint saved yet
            return True
        
        return ((self.maximize_metric and self.best_val < metric_val)
                or (not self.maximize_metric and self.best_val > metric_val))
    
    def save(self, step, model, metric_val, device):
        """Save model parameters to disk.

        Args:
            step (int): Total number of examples seen during training so far.
            model (torch.nn.DataParallel): Model to save.
            metric_val (float): Determines whether checkpoint is best so far.
            device (torch.device): Device where model resides.
        """

        ckpt_dict = {
            'model_name': model.__class__.__name__,
            'mode_state': model.cpu().state_dict(),
            'step': step
        }

        model.to(device)

        checkpoint_path = os.path.join(self.save_dir, f'step_{step}.pth.tar')
        torch.save(ckpt_dict, checkpoint_path)
        print(f'Saved checkpoint: {checkpoint_path}')

        if self.is_best(metric_val):
            # Save the best model
            self.best_val = metric_val
            best_path = os.path.join(self.save_dir, 'best.pth.tar')
            shutil.copy(checkpoint_path, best_path)
            print(f'New best checkpoint at step {step}...')
        
        # Add checkpoint path to priority queue (lowest priority removed first)
        if self.maximize_metric:
            priority_order = metric_val
        else:
            priority_order = -metric_val

        self.ckpt_paths.put((priority_order, checkpoint_path))

        # Remove a checkpoint if more than max_checkpoints have been saved
        if self.ckpt_paths.qsize() > self.max_checkpoints:
            _, worst_ckpt = self.ckpt_paths.get()
            try:
                os.remove(worst_ckpt)
                print(f'Removed checkpoint: {worst_ckpt}')
            except OSError:
                # Avoid crashing if checkpoint has been removed or protected
                pass


class AverageMeter:
    """Keep track of average values over time.

    Adapted from:
        > https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def reset(self):
        """Reset meter."""
        self.__init__()
    
    def update(self, val, num_samples=1):
        """Update meter with new value `val`, the average of `num` samples.
        
        Args:
            val (float): Average value to updater the meter with.
            num_samples (int): Number of samples that were averaged to produce `val`.
        """
        self.count += num_samples
        self.sum += val * num_samples
        self.avg = self.sum / self.count

def torch_from_json(path, dtype=torch.float32):
    """Load a PyTorch Tensor from a JSON file.

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


def get_save_dir(base_dir, name, training, id_max=100):
    """Get a unique save directory by appending the smallest positive integer
    `id < id_max` that is not already taken (i.e., no dir exists with that id).
    
    Args:
        base_dir (str): Base directory in which to make save directories.
        name (str): Name to identify this training run.
        training (bool): Whether save directory is for training (false => test)
        id_max (int): Maximum ID number before raising an exception.
    
    Returns:
        save_dir (str): Path to a new directory with a unique name.    
    """

    for uid in range(1, id_max):
        subdir = 'train' if training else 'test'
        save_dir = os.path.join(base_dir, subdir, f'{name}-{uid:02d}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir
    
    raise RuntimeError('Too many save directories created with the same name. \
                        Delete old save directories or use another name.')

def get_available_devices():
    """Get IDs of all available GPUs.

    Returns:
        device (torch.device): Main device (GPU 0 or CPU)
        gpu_ids (list): List of IDs of all GPUs that are available.
    """

    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    
    return device, gpu_ids

def load_model(model, checkpoint_path, gpu_ids, return_step=True):
    """Load model parameters from disk.

    Args:
        model (torch.nn.DataParallel): Load parameters into this model.
        checkpoint_path (str): Path to checkpoint to load.
        gpu_ids (list): GPU IDs for DataParallel.
        return_step (bool): Also return the step at which checkpoint was saved.
    
    Returns:
        model (torch.nn.DataParallel): Model loaded from checkpoint.
        step (int): Step at which checkpoint was saved. Only if `return_step`.
    """

    device = f"cuda:{gpu_ids[0]}" if gpu_ids else 'cpu'
    ckpt_dict = torch.load(checkpoint_path, map_location=device)

    # Build model, load parameters
    model.load_state_dict(ckpt_dict['model_state'])

    if return_step:
        step = ckpt_dict['step']
        return model, step

    return model