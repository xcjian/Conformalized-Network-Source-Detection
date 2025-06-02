
import torch
import torch.nn as nn
import torch.nn.functional as F

def gconv(x, theta, Ks, c_in, c_out, Lk):
    batch_time, N, _ = x.shape
    x_g = []
    for k in range(Ks):
        T_k = Lk[k].to(x.device)  # [N, N] tensor
        x1 = torch.einsum("ij,bjc->bic", T_k, x)
        x_g.append(x1)
    x_g = torch.cat(x_g, dim=-1)  # [B*T, N, Ks*c_in]
    x_theta = torch.matmul(x_g, theta)  # [B*T, N, c_out]
    return x_theta

def gcn(x, Ks, c_out, Lk):
    # simplified gcn: only first-order approximation assumed
    # Lk[0] assumed to be the normalized adjacency matrix
    T_k = Lk[0].to(x.device) 
    return torch.einsum("ij,bjc->bic", T_k, x)

def layer_norm(x):
    '''
    Layer normalization function.
    x: tensor, [batch_size, time_step, N, channel].
    scope: str, variable scope.
    return: tensor, [batch_size, time_step, N, channel].
    '''
    _, _, N, C = x.shape
    mu = x.mean(dim=(2, 3), keepdim=True)
    sigma = x.var(dim=(2, 3), keepdim=True, unbiased=False)
    gamma = torch.ones(1, 1, N, C, device=x.device)
    beta = torch.zeros(1, 1, N, C, device=x.device)
    _x = (x - mu) / torch.sqrt(sigma + 1e-6) * gamma + beta
    return _x


class TemporalConv(nn.Module):
    def __init__(self, Kt, c_in, c_out, act_func='relu'):
        super().__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.act_func = act_func

        if act_func == 'GLU':
            self.conv = nn.Conv2d(c_in, 2 * c_out, kernel_size=(Kt, 1))
        else:
            self.conv = nn.Conv2d(c_in, c_out, kernel_size=(Kt, 1))

        if c_in > c_out:
            self.downsample = nn.Conv2d(c_in, c_out, kernel_size=1)
        else:
            self.downsample = None

    def forward(self, x):
        B, T, N, _ = x.shape
        x = x.permute(0, 3, 1, 2)  # [B, C_in, T, N]
        if self.downsample:
            x_input = self.downsample(x)
        elif self.c_in < self.c_out:
            pad = torch.zeros(x.size(0), self.c_out - self.c_in, x.size(2), x.size(3), device=x.device)
            x_input = torch.cat([x, pad], dim=1)
        else:
            x_input = x

        x_input = x_input[:, :, self.Kt - 1:T, :]
        x_conv = self.conv(x)

        if self.act_func == 'GLU':
            x_conv1, x_conv2 = x_conv.chunk(2, dim=1)
            out = (x_conv1 + x_input) * torch.sigmoid(x_conv2)
        elif self.act_func == 'linear':
            out = x_conv
        elif self.act_func == 'sigmoid':
            out = torch.sigmoid(x_conv)
        elif self.act_func == 'relu':
            out = F.relu(x_conv + x_input)
        else:
            raise ValueError(f'Unknown activation function: {self.act_func}')

        return out.permute(0, 2, 3, 1)

# class TemporalConv(nn.Module):
#     def __init__(self, Kt, c_in, c_out, act_func='relu'):
#         super().__init__()
#         self.Kt = Kt
#         self.c_in = c_in
#         self.c_out = c_out
#         self.act_func = act_func

#         out_channels = 2 * c_out if act_func == 'GLU' else c_out

#         # ➤ no padding here!
#         self.conv = nn.Conv2d(c_in, out_channels, kernel_size=(Kt, 1), padding=0)
#         self.res_proj = nn.Conv2d(c_in, c_out, kernel_size=(1, 1)) if c_in != c_out else nn.Identity()

#     def forward(self, x):
#         x = x.permute(0, 3, 1, 2).contiguous()  # [B, C, T, N]

#         # ➤ causal padding done here manually, once
#         x_padded = F.pad(x, (0, 0, self.Kt - 1, 0))  # pad time dim left

#         x_conv = self.conv(x_padded)
#         x_res = self.res_proj(x)

#         # ➤ ensure alignment: crop residual to match x_conv
#         T_conv = x_conv.shape[2]
#         x_res = x_res[:, :, -T_conv:, :]  # align to causal output

#         if self.act_func == 'GLU':
#             x_conv1, x_conv2 = x_conv.chunk(2, dim=1)
#             out = (x_conv1 + x_res) * torch.sigmoid(x_conv2)
#         elif self.act_func == 'linear':
#             out = x_conv
#         elif self.act_func == 'sigmoid':
#             out = torch.sigmoid(x_conv)
#         elif self.act_func == 'relu':
#             out = F.relu(x_conv + x_res)
#         else:
#             raise ValueError(f'Unknown activation function: {self.act_func}')

#         return out.permute(0, 2, 3, 1).contiguous()  # [B, T, N, C_out]







class SpacialConvCheb(nn.Module):
    def __init__(self, Ks, c_in, c_out, Lk):
        super().__init__()
        self.Ks = Ks
        self.c_in = c_in
        self.c_out = c_out
        self.Lk = Lk
        self.ws = nn.Parameter(torch.randn(Ks * c_in, c_out))
        self.bs = nn.Parameter(torch.zeros(c_out))
        if c_in > c_out:
            self.downsample = nn.Conv2d(c_in, c_out, kernel_size=1)
        else:
            self.downsample = None

    def forward(self, x):
        B, T, N, _ = x.shape
        if self.downsample:
            x_input = self.downsample(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        elif self.c_in < self.c_out:
            pad = torch.zeros(B, T, N, self.c_out - self.c_in, device=x.device)
            x_input = torch.cat([x, pad], dim=-1)
        else:
            x_input = x
        x_reshape = x_input.reshape(-1, N, self.c_in)
        x_gconv = gconv(x_reshape, self.ws, self.Ks, self.c_in, self.c_out, self.Lk) + self.bs
        x_gc = x_gconv.view(B, T, N, self.c_out)
        return F.relu(x_gc + x_input[:, :, :, :self.c_out])
    

# class SpacialConvGCN(nn.Module):
#     def __init__(self, Ks, c_in, c_out, Lk):
#         '''
#         Ks: spatial kernel size (for GCN, just used as a dummy)
#         c_in: input channels
#         c_out: output channels
#         Lk: list of graph kernel matrices (expected Lk[0] = A_hat)
#         '''
#         super().__init__()
#         self.Ks = Ks
#         self.c_in = c_in
#         self.c_out = c_out
#         self.Lk = Lk  # Lk[0] should be [N, N] adjacency matrix

#         # Linear projection layer: applies to last channel dim
#         self.linear = nn.Linear(c_in, c_out)

#         # Optional downsampling for residual
#         if c_in > c_out:
#             self.downsample = nn.Conv2d(c_in, c_out, kernel_size=1)
#         else:
#             self.downsample = None

#     def forward(self, x):
#         '''
#         x: [B, T, N, C]
#         returns: [B, T, N, C_out]
#         '''
#         B, T, N, C = x.shape

#         # Residual input preparation
#         if self.downsample:
#             x_input = self.downsample(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # [B, T, N, C_out]
#         elif self.c_in < self.c_out:
#             pad = torch.zeros(B, T, N, self.c_out - self.c_in, device=x.device)
#             x_input = torch.cat([x, pad], dim=-1)
#         else:
#             x_input = x

#         # GCN propagation: A_hat @ x
#         A_hat = self.Lk[0].to(x.device)              # [N, N]
#         x_prop = torch.einsum("ij,btjc->btic", A_hat, x)  # [B, T, N, C_in]

#         # Linear projection over last dim: C_in → C_out
#         x_proj = self.linear(x_prop)  # [B, T, N, C_out]

#         # Residual connection + activation
#         return F.relu(x_proj + x_input[:, :, :, :self.c_out])

class SpacialConvGCN(nn.Module):
    def __init__(self, Ks, c_in, c_out, Lk):
        super().__init__()
        self.Ks = Ks
        self.c_in = c_in
        self.c_out = c_out
        self.Lk = Lk  # [A_hat]

        self.gcn_layers = nn.ModuleList([
            nn.Linear(c_in, c_out) for _ in range(Ks)
        ])

        self.downsample = nn.Conv2d(c_in, c_out, kernel_size=1) if c_in > c_out else None

    def forward(self, x):
        B, T, N, C = x.shape
        A_hat = self.Lk[0].to(x.device)  # [N, N]
        if self.downsample:
            x_input = self.downsample(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        elif self.c_in < self.c_out:
            pad = torch.zeros(B, T, N, self.c_out - self.c_in, device=x.device)
            x_input = torch.cat([x, pad], dim=-1)
        else:
            x_input = x

        out = 0
        for k in range(self.Ks):
            x_k = torch.einsum("ij,btjc->btic", A_hat, x)  # A_hat @ x
            out += F.relu(self.gcn_layers[k](x_k))  # Linear + ReLU

        return F.relu(out + x_input[:, :, :, :self.c_out])



class STConvBlock(nn.Module):
    '''
    Spatio-temporal convolutional block, which contains two temporal gated convolution layers
    and one spatial graph convolution layer in the middle.
    x: tensor, batch_size, time_step, N, c_in].
    Ks: int, kernel size of spatial convolution.
    Kt: int, kernel size of temporal convolution.
    channels: list, channel configs of a single st_conv block.
    scope: str, variable scope.
    keep_prob: placeholder, prob of dropout.
    sconv: type of spatio-convolution, cheb or gcn
    act_func: str, activation function.
    return: tensor, [batch_size, time_step, N, c_out].
    '''
    def __init__(self, Ks, Kt, channels, keep_prob, sconv, Lk, act_func='GLU', n_node=None):
        super().__init__()
        c_si, c_t, c_oo = channels
        self.temporal1 = TemporalConv(Kt, c_si, c_t, act_func=act_func)
        if sconv == 'cheb':
            self.spatial = SpacialConvCheb(Ks, c_t, c_t, Lk)
        elif sconv == 'gcn':
            self.spatial = SpacialConvGCN(Ks, c_t, c_t, Lk)
        else:
            raise ValueError(f"Unknown spatio-conv method: {sconv}")
        self.temporal2 = TemporalConv(Kt, c_t, c_oo)
        self.keep_prob = keep_prob
        self.n_node = n_node or 774
        self.layer_norm = nn.LayerNorm([self.n_node, c_oo])


    def forward(self, x):
        x = self.temporal1(x)
        x = self.spatial(x)
        x = self.temporal2(x)
        x = self.layer_norm(x)

        return F.dropout(x, p=1 - self.keep_prob)

class FullyConv(nn.Module):
    def __init__(self, n, channel, scope=None):
        '''
        Fully connected layer: maps multi-channels to one.
        :param n: int, number of nodes / size of graph (not used).
        :param channel: int, number of input channels.
        :param scope: str, unused in PyTorch version but kept for compatibility.
        '''
        super(FullyConv, self).__init__()
        self.fc = nn.Conv2d(channel, 1, kernel_size=1)

    def forward(self, x):
        '''
        x: tensor, [batch_size, 1, N, channel]
        return: tensor, [batch_size, 1, N, 1]
        '''
        x = x.permute(0, 3, 1, 2)       # [B, C, 1, N]
        x_fc = self.fc(x)               # [B, 1, 1, N]
        x_fc = x_fc.permute(0, 2, 3, 1) # [B, 1, N, 1]
        return x_fc


class FullyConv_nodewise(nn.Module):
    def __init__(self, n, out_channels, in_channels=None, scope=None):
        '''
        Fully connected layer: maps multi-channels to multiple outputs per node.
        n: int, number of nodes.
        out_channels: int, number of output channels (e.g., 2).
        in_channels: int, number of input channels. (optional if known dynamically)
        scope: unused, kept for compatibility.
        '''
        super(FullyConv_nodewise, self).__init__()
        self.out_channels = out_channels
        self.n = n
        self.scope = scope

        # Placeholder for Conv2d, to be built on first forward if in_channels not provided
        self.fc = None
        self.in_channels = in_channels

        # Bias: [N, out_channels], register as parameter
        self.b = nn.Parameter(torch.zeros(n, out_channels))  # [N, C_out]

    def build_fc(self, in_channels):
        self.in_channels = in_channels
        self.fc = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)

    def forward(self, x):
        '''
        x: [batch_size, 1, N, in_channels]
        return: [batch_size, 1, N, out_channels]
        '''
        B, _, N, C_in = x.shape
        if self.fc is None:
            self.build_fc(C_in)

        # [B, 1, N, C_in] → [B, C_in, 1, N]
        x_ = x.permute(0, 3, 1, 2)
        x_out = self.fc(x_)  # → [B, C_out, 1, N]
        x_out = x_out.permute(0, 2, 3, 1)  # → [B, 1, N, C_out]

        # Add bias [N, C_out] → [1, 1, N, C_out]
        x_out = x_out + self.b.unsqueeze(0).unsqueeze(0)

        return x_out


class OutputLayer(nn.Module):
    def __init__(self, T, C, act_func='GLU'):
        super().__init__()
        self.temporal1 = TemporalConv(T, C, C, act_func=act_func)
        self.temporal2 = TemporalConv(1, C, C, act_func='sigmoid')
        self.fc = nn.Conv2d(C, 1, kernel_size=1)

    def forward(self, x):
        x = self.temporal1(x)
        x = layer_norm(x)
        x = self.temporal2(x)
        x_fc = self.fc(x.permute(0, 3, 1, 2))  # [B, 1, T, N]
        x_fc = x_fc.permute(0, 2, 3, 1)        # [B, T, N, 1]
        return x_fc[:, 0, :, 0]                # [B, N]

class OutputLayer_nodewise(nn.Module):
    def __init__(self, T, n_node, channel, act_func='GLU'):
        '''
        Output layer: temporal convolution layers + FCs for nodewise 2-class classification.
        :param T: int, temporal kernel size for first conv
        :param n_node: int, number of nodes
        :param channel: int, input & hidden channel size
        :param act_func: activation function, e.g., 'GLU'
        '''
        super(OutputLayer_nodewise, self).__init__()
        self.temporal_in = TemporalConv(T, channel, channel, act_func)
        self.temporal_out = TemporalConv(1, channel, channel, act_func='sigmoid')
        self.layer_norm = nn.LayerNorm([n_node, channel])
        self.fc0 = FullyConv(n_node, channel)
        self.fc1 = FullyConv(n_node, channel)

    def forward(self, x):
        '''
        :param x: tensor, shape [B, T, N, C]
        :return: tensor, shape [B, N, 2]
        '''
        x_i = self.temporal_in(x)                      # [B, T', N, C]
        x_ln = self.layer_norm(x_i)              # [B, T', N, C]
        x_o = self.temporal_out(x_ln)                  # [B, T'', N, C]
        x_o = torch.mean(x_o, dim=1, keepdim=True)     # [B, 1, N, C]
        out0 = self.fc0(x_o)                           # [B, 1, N, 1]
        out1 = self.fc1(x_o)                           # [B, 1, N, 1]
        logits = torch.cat([out0, out1], dim=-1)       # [B, 1, N, 2]
        logits = logits.squeeze(1)                     # [B, N, 2]
        return logits
