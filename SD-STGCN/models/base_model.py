from models.layers import *
from os.path import join as pjoin
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class STGCN_SI_Nodewise(nn.Module):
    def __init__(self, n_frame, Ks, Kt, blocks, sconv, Lk, keep_prob, pos_weight=1.0, n_node=None):
        super(STGCN_SI_Nodewise, self).__init__()
        """
        STGCN-SI with nodewise prediction.
        """
        self.n_frame = n_frame
        self.Ks = Ks
        self.Kt = Kt
        self.blocks = blocks
        self.sconv = sconv
        self.Lk = Lk
        self.keep_prob = keep_prob
        self.pos_weight = pos_weight
        self.n_node = n_node or 774  # default node count

        self.st_blocks = nn.ModuleList([
            STConvBlock(Ks, Kt, channels, keep_prob, sconv, Lk, act_func='GLU')
            for channels in blocks
        ])

        Ko = n_frame - 2 * (Kt - 1) * len(blocks)
        if Ko <= 1:
            raise ValueError(f"ERROR: kernel size Ko must be > 1, but received {Ko}.")

        last_channels = blocks[-1][-1]
        self.output = OutputLayer_nodewise(Ko, self.n_node, last_channels)

    def forward(self, x, y=None, training=True):
        """
        x: [B, T, N, C]
        y: [B, N], 0 or 1 per node, optional
        training: bool, if True return (logits, loss), else only logits
        """
        x_infect = x[:, 0, :, 1]  # infection status at t=0, ch=1 → [B, N]

        for block in self.st_blocks:
            x = block(x)

        logits = self.output(x)  # [B, N, 2]
        if training:
            if y is None:
                raise ValueError("Ground-truth labels y must be provided during training.")

            B, N = y.shape
            logits_flat = logits.view(-1, 2)         # [B*N, 2]
            y_flat = y.view(-1).long()               # [B*N]

            infected_mask = (x_infect > 0).float().view(-1)  # [B*N]
            valid_indices = infected_mask.nonzero(as_tuple=False).squeeze()

            if valid_indices.numel() == 0:
                loss = torch.tensor(0.0, device=x.device, requires_grad=True)
                return logits, loss

            logits_valid = logits_flat[valid_indices]  # [num_infected, 2]
            y_valid = y_flat[valid_indices]            # [num_infected]

            weights = torch.tensor([1.0, self.pos_weight], device=x.device)
            loss = F.cross_entropy(logits_valid, y_valid, weight=weights)

            return logits, loss
        else:
            return logits

def model_save(model, global_step, model_name, save_path='./output/models/'):
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"{model_name}_step{global_step}_pytorch.pt")
    torch.save(model.state_dict(), save_file)
    print(f"<< Saving model to {save_file} ...")