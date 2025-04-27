# import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

def gconv(x, theta, Ks, c_in, c_out, Lk):
    batch_time, N, _ = x.shape
    x_g = []
    for k in range(Ks):
        T_k = Lk[k]  # [N, N] tensor
        x1 = torch.einsum("ij,bjc->bic", T_k, x)
        x_g.append(x1)
    x_g = torch.cat(x_g, dim=-1)  # [B*T, N, Ks*c_in]
    x_theta = torch.matmul(x_g, theta)  # [B*T, N, c_out]
    return x_theta


def gcn(x, Ks, c_out, Lk):
    # simplified gcn: only first-order approximation assumed
    # Lk[0] assumed to be the normalized adjacency matrix
    T_k = Lk[0]  # [N, N]
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


class temporal_conv_layer(nn.Module):
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


class spatio_conv_layer_cheb(nn.Module):
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


class spatio_conv_layer_gcn(nn.Module):
    def __init__(self, Ks, c_in, c_out, Lk):
        super().__init__()
        self.Ks = Ks
        self.c_in = c_in
        self.c_out = c_out
        self.Lk = Lk
        if c_in > c_out:
            self.downsample = nn.Conv2d(c_in, c_out, kernel_size=1)
        else:
            self.downsample = None
        self.linear = nn.Linear(c_in, c_out)


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
        x_gconv = gcn(x_reshape, self.Ks, self.c_out, self.Lk)
        x_gconv = self.linear(x_gconv)
        x_gc = x_gconv.view(B, T, N, self.c_out)
        return F.relu(x_gc + x_input[:, :, :, :self.c_out])
    

class st_conv_block(nn.Module):
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
    def __init__(self, Ks, Kt, channels, keep_prob, sconv, Lk, act_func='GLU'):
        super().__init__()
        c_si, c_t, c_oo = channels
        self.temporal1 = temporal_conv_layer(Kt, c_si, c_t, act_func=act_func)
        if sconv == 'cheb':
            self.spatial = spatio_conv_layer_cheb(Ks, c_t, c_t, Lk)
        elif sconv == 'gcn':
            self.spatial = spatio_conv_layer_gcn(Ks, c_t, c_t, Lk)
        else:
            raise ValueError(f"Unknown spatio-conv method: {sconv}")
        self.temporal2 = temporal_conv_layer(Kt, c_t, c_oo)
        self.keep_prob = keep_prob

    def forward(self, x):
        x = self.temporal1(x)
        x = self.spatial(x)
        x = self.temporal2(x)
        x = layer_norm(x)

        return F.dropout(x, p=1 - self.keep_prob)
    

def fully_con_layer(x, n, channel, scope):
    '''
    Fully connected layer: maps multi-channels to one.
    x: tensor, [batch_size, 1, N, channel].
    n: int, number of nodes / size of graph.
    channel: channel size of input x.
    scope: str, variable scope.
    return: tensor, [batch_size, 1, N, 1].
    '''

    fc = nn.Conv2d(channel, 1, kernel_size=1).to(x.device)
    x_fc = fc(x_o.permute(0, 3, 1, 2))  # [B, 1, T, N]
    x_fc = x_fc.permute(0, 2, 3, 1)  # [B, T, N, 1]
    return x_fc

class output_layer(nn.Module):
    def __init__(self, T, C, act_func='GLU'):
        super().__init__()
        self.temporal1 = temporal_conv_layer(T, C, C, act_func=act_func)
        self.temporal2 = temporal_conv_layer(1, C, C, act_func='sigmoid')
        self.fc = nn.Conv2d(C, 1, kernel_size=1)

    def forward(self, x):
        x = self.temporal1(x)
        x = layer_norm(x)
        x = self.temporal2(x)
        x_fc = self.fc(x.permute(0, 3, 1, 2))  # [B, 1, T, N]
        x_fc = x_fc.permute(0, 2, 3, 1)        # [B, T, N, 1]
        return x_fc[:, 0, :, 0]                # [B, N]


def variable_summaries(var, v_name):
    '''
    Attach summaries to a Tensor (for TensorBoard visualization).
    Ref: https://zhuanlan.zhihu.com/p/33178205
    var: tf.Variable().
    v_name: str, name of the variable.
    '''
    with tf.compat.v1.name_scope('summaries'):
        mean = tf.reduce_mean(input_tensor=var)
        tf.compat.v1.summary.scalar(f'mean_{v_name}', mean)

        with tf.compat.v1.name_scope(f'stddev_{v_name}'):
            stddev = tf.sqrt(tf.reduce_mean(input_tensor=tf.square(var - mean)))
        tf.compat.v1.summary.scalar(f'stddev_{v_name}', stddev)

        tf.compat.v1.summary.scalar(f'max_{v_name}', tf.reduce_max(input_tensor=var))
        tf.compat.v1.summary.scalar(f'min_{v_name}', tf.reduce_min(input_tensor=var))

        tf.compat.v1.summary.histogram(f'histogram_{v_name}', var)
