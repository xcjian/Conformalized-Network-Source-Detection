from data_loader.data_utils_pyt import *
from utils.metric_utils import *
from models_pyt.base_model import STGCN_SI, model_save

import numpy as np
import time
import os

import torch
import torch.nn.functional as F

def model_train(inputs, blocks, args, Lk, save_path='./output/models/', sum_path='./output/tensorboard'):
    '''
    Train the base model.
    inputs: instance of class Dataset, data source for training.
    blocks: list, channel configs of st_conv blocks.
    args: instance of class argparse, args for training.
    '''
    n, n_frame, n_channel = args.n_node, args.n_frame, args.n_channel
    Ks, Kt = args.ks, args.kt
    batch_size, epoch, opt = args.batch_size, args.epoch, args.opt
    sconv = args.sconv # spatio-conv type
    dropout = args.dropout
    start, end = args.start, args.end
    valid = args.valid
    random = args.random

    device = args.device
    model = STGCN_SI(n_frame, Ks, Kt, blocks, sconv, Lk, keep_prob=1 - dropout).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    if opt == 'RMSProp':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif opt == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f'ERROR: optimizer "{opt}" is not defined.')
    
    len_train = inputs.get_len('train')
    epoch_step = len_train // batch_size + (1 if len_train % batch_size != 0 else 0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5 * epoch_step, gamma=0.7)

    # Initialize variables to track the best validation accuracy and corresponding model
    best_val_acc = 0.0
    best_epoch = 0

    for epoch_idx in range(epoch):
        start_time = time.time()
        model.train()

        for j, (x_batch, y_batch) in enumerate(
            gen_xy_batch(inputs.get_data('train'), batch_size, dynamic_batch=True, shuffle=True)):

            x_tensor = onehot(iteration2snapshot(x_batch, n_frame, start=start, end=end, random=random), n_channel)
            x_tensor = torch.tensor(x_tensor).float().to(device)
            y_tensor = torch.tensor(y_batch).long().to(device)

            optimizer.zero_grad()
            logits = model(x_tensor)
            loss = loss_fn(logits, y_tensor)
            loss.backward()
            optimizer.step()

            if j % 10 == 0:
                acc = batch_acc(logits.detach().cpu().numpy(), y_batch)

                if valid:
                    model.eval()
                    acc_val_list = []
                    with torch.no_grad():
                        for (x_val, y_val) in gen_xy_batch(inputs.get_data('val'), batch_size, dynamic_batch=True, shuffle=False):
                            x_val_ = onehot(iteration2snapshot(x_val, n_frame, start=start, end=end, random=random), n_channel)
                            x_val_tensor = torch.tensor(x_val_).float().to(device)
                            y_val_tensor = torch.tensor(y_val).long().to(device)

                            val_logits = model(x_val_tensor)
                            acc_val_list.append(batch_acc(val_logits.cpu().numpy(), y_val))

                    acc_val = np.mean(acc_val_list)
                    end_time = time.time()
                    print(f'Epoch {epoch_idx:2d}, Step {j:3d}: Elapsed Time {end_time-start_time:.3f} Loss {loss.item():.3f} Train Acc {acc:.3f} Val Acc {acc_val:.3f}')

                    if acc_val > best_val_acc:
                        best_val_acc = acc_val
                        best_epoch = epoch_idx
                        for file in os.listdir(save_path):
                            if file.startswith('STGCN'):
                                os.remove(os.path.join(save_path, file))
                        model_save(model, epoch_idx, 'STGCN', save_path)
                        print(f'New best model saved with validation accuracy: {best_val_acc:.3f} at epoch {best_epoch}')
                else:
                    end_time = time.time()
                    print(f'Epoch {epoch_idx:2d}, Step {j:3d}: Elapsed Time {end_time-start_time:.3f} Loss {loss.item():.3f} Train Acc {acc:.3f}')

        
        scheduler.step()

    if valid:
        print(f'Final model saved with best validation accuracy: {best_val_acc:.3f}')
    else:
        model_save(model, epoch_idx, 'STGCN', save_path)
        print('Final model saved.')

    print('Training model finished!')

