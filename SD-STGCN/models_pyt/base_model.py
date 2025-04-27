from models_pyt.layers import *
from os.path import join as pjoin
# import tensorflow as tf
import torch
import torch.nn as nn
# import torch.nn.functional as F
import os

def build_model(x, y, n_frame, Ks, Kt, blocks, keep_prob, sconv):
    '''
    Build the base model.
    x: placeholder features, [-1, n_frame, n, n_channel]
    y: placeholder label, [-1, n]
    n_frame: int, size of records for training.
    Ks: int, kernel size of spatial convolution.
    Kt: int, kernel size of temporal convolution.
    blocks: list, channel configs of st_conv blocks.
    keep_prob: placeholder.
    sconv: type of spatio-convolution layer, cheb or gcn
    '''

    # Ko>0: kernel size of temporal convolution in the output layer.
    Ko = n_frame

    # ST-Block
    for i, channels in enumerate(blocks):
        x = st_conv_block(x, Ks, Kt, channels, i, keep_prob, sconv, act_func='GLU')
        Ko -= 2 * (Kt - 1)

    # Output Layer
    if Ko > 1:
        # logits shape: [-1, n_node]
        logits = output_layer(x, Ko, 'output_layer')
    else:
        raise ValueError(f'ERROR: kernel size Ko must be greater than 1, but received "{Ko}".')



    train_loss = tf.compat.v1.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
                                            labels=y, logits=logits, axis=-1))


    y_pred = tf.nn.softmax(logits)
    tf.compat.v1.add_to_collection(name='y_pred', value=y_pred)

    return train_loss, y_pred


class STGCN_SI(nn.Module):
    def __init__(self, n_frame, Ks, Kt, blocks, sconv, Lk, keep_prob):
        super(STGCN_SI, self).__init__()
        self.n_frame = n_frame
        self.Ks = Ks
        self.Kt = Kt
        self.blocks = blocks
        self.sconv = sconv
        self.Lk = Lk
        self.keep_prob = keep_prob

        self.st_blocks = nn.ModuleList([
            st_conv_block(Ks, Kt, channels, keep_prob, sconv, Lk, act_func='GLU')
            for channels in blocks
        ])

        Ko = n_frame - 2 * (Kt - 1) * len(blocks)
        if Ko <= 1:
            raise ValueError(f"ERROR: kernel size Ko must be > 1, but received {Ko}.")
        
        last_channels = blocks[-1][-1]  # c_oo of last block
        self.output = output_layer(Ko, last_channels)

    def forward(self, x):
        for block in self.st_blocks:
            x = block(x)

        logits = self.output(x)

        # Apply infection mask
        infection_status = x[:, 0, :, 1]  # Select the second feature at time step 0
        infected_mask = (infection_status > 0).float()
        logits = logits * infected_mask + (1 - infected_mask) * (-1e9)

        return logits
    

def model_save(model, global_step, model_name, save_path='./output/models/'):
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"{model_name}_step{global_step}.pt")
    torch.save(model.state_dict(), save_file)
    print(f"<< Saving model to {save_file} ...")