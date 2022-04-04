"""Train a language model
Adapted from:
    https://github.com/michiyasunaga/squad/blob/main/train.py
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import util

from args import get_train_args
from collections import OrderedDict
from json import dumps
from models import RNNModule
from tqdm import tqdm
from ujson import load as json_load
from util import TextDataset

def main(args):
    # Set up logging and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=True)
    device, args.gpu_ids = util.get_available_devices()
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seed
    print(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get embeddings
    print('Loading embeddings...')
    word_vectors = util.torch_from_json(args.word_emb_file)

    # Get model
    model = RNNModule(word_vectors, args.hidden_size)
    model = nn.DataParallel(model, args.gpu_ids)
    if args.load_path:
        print(f'Loading checkpoint from {args.load_path}...')
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0
    
    model.to(device)
    model.train()

    # Get saver
    saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric)

    # Get optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                weight_decay=args.l2_wd)
    scheduler = sched.LambdaLR(optimizer, lambda s: 1.) # Constant LR

    # Get data loader
    print('Getting dataset...')
    train_dataset = TextDataset(args.train_features_file, args.sequence_length)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=args.num_workers)
    dev_dataset = TextDataset(args.dev_features_file, args.sequence_length)
    dev_loader = data.DataLoader(dev_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.num_workers)
    
    # Train
    print('Training...')
    steps_till_eval = args.eval_steps
    epoch = step // len(train_dataset)
    while epoch != args.num_epochs:
        epoch += 1
        print(f'Starting epoch {epoch}...')        
        state_h, state_c = model.init_state(args.batch_size)
        state_h = state_h.to(device)
        state_c = state_c.to(device)

        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:
            for x, y in train_loader:                
                # Setup for forward
                x = x.to(device)
                y = y.to(device)
                batch_size = x.size(0)
                state_h = state_h[:,:batch_size,:]
                state_c = state_c[:,:batch_size,:]
                optimizer.zero_grad()
                
                # Forward
                output, (state_h, state_c) = model(x, (state_h, state_c))
                state_h = state_h.detach()
                state_c = state_c.detach()
                loss = F.cross_entropy(output.transpose(1,2), y)
                loss_val = loss.item()

                # Backward                
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch, loss=loss_val)

                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    print(f'Evaluating at step {step}')
                    # To be continued...


def evaluate(model, data_loader, device):
    loss_meter = util.AverageMeter()
    ppl_meter = util.AverageMeter()

    model.eval()
    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for x, y in data_loader:
            # Setup for forward
            x = x.to(device)
            y = y.to(device)
            batch_size = x.size(0)
            state_h = state_h[:,:batch_size,:]
            state_c = state_c[:,:batch_size,:]

            # Forward
            output, _ = model(x, state_h, state_c)
            loss = F.cross_entropy(output.transpose(1,2), y)
            loss_meter.update(loss.item(), batch_size)
            
            # Calculate perplexity
            ppl = loss.exp()
            ppl_meter.update(ppl.item(), batch_size)

            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(loss=loss_meter.avg, ppl=ppl_meter.avg)
    
    model.train()

    results_list = [('Loss', loss_meter.avg),
                    ('Perplexity', ppl_meter.avg)]
    results = OrderedDict(results_list)

    return results

if __name__ == '__main__':
    main(get_train_args())
