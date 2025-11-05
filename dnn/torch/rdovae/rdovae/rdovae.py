"""
/* Copyright (c) 2022 Amazon
   Written by Jan Buethe */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
"""

""" Pytorch implementations of rate distortion optimized variational autoencoder """

import math as m

import torch
from torch import nn
import torch.nn.functional as F
import sys
import os
import math
source_dir = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(os.path.join(source_dir, "../../lpcnet/"))
#from utils.sparsification import GRUSparsifier
from torch.nn.utils import weight_norm
sys.path.append(os.path.join(source_dir, "../../dnntools"))
from dnntools.quantization import soft_quant
from dnntools.sparsification import GRUSparsifier, LinearSparsifier

# Quantization and rate related utily functions

def soft_pvq(x, k):
    """ soft pyramid vector quantizer """

    # L2 normalization
    x_norm2 = x / (1e-15 + torch.norm(x, dim=-1, keepdim=True))


    with torch.no_grad():
        # quantization loop, no need to track gradients here
        x_norm1 = x / torch.sum(torch.abs(x), dim=-1, keepdim=True)

        # set initial scaling factor to k
        scale_factor = k
        x_scaled = scale_factor * x_norm1
        x_quant = torch.round(x_scaled)

        # we aim for ||x_quant||_L1 = k
        for _ in range(10):
            # remove signs and calculate L1 norm
            abs_x_quant = torch.abs(x_quant)
            abs_x_scaled = torch.abs(x_scaled)
            l1_x_quant = torch.sum(abs_x_quant, axis=-1)

            # increase, where target is too small and decrease, where target is too large
            plus  = 1.0001 * torch.min((abs_x_quant + 0.5) / (abs_x_scaled + 1e-15), dim=-1).values
            minus = 0.9999 * torch.max((abs_x_quant - 0.5) / (abs_x_scaled + 1e-15), dim=-1).values
            factor = torch.where(l1_x_quant > k, minus, plus)
            factor = torch.where(l1_x_quant == k, torch.ones_like(factor), factor)
            scale_factor = scale_factor * factor.unsqueeze(-1)

            # update x
            x_scaled = scale_factor * x_norm1
            x_quant = torch.round(x_quant)

    # L2 normalization of quantized x
    x_quant_norm2 = x_quant / (1e-15 + torch.norm(x_quant, dim=-1, keepdim=True))
    quantization_error = x_quant_norm2 - x_norm2

    return x_norm2 + quantization_error.detach()

def cache_parameters(func):
    cache = dict()
    def cached_func(*args):
        if args in cache:
            return cache[args]
        else:
            cache[args] = func(*args)

        return cache[args]
    return cached_func

@cache_parameters
def pvq_codebook_size(n, k):

    if k == 0:
        return 1

    if n == 0:
        return 0

    return pvq_codebook_size(n - 1, k) + pvq_codebook_size(n, k - 1) + pvq_codebook_size(n - 1, k - 1)


def soft_rate_estimate(z, r, reduce=True):
    """ rate approximation with dependent theta Eq. (7)"""

    rate = torch.sum(
        - torch.log2((1 - r)/(1 + r) * r ** torch.abs(z) + 1e-6),
        dim=-1
    )

    if reduce:
        rate = torch.mean(rate)

    return rate


def hard_rate_estimate(z, r, theta, reduce=True):
    """ hard rate approximation """

    z_q = torch.round(z)
    p0 = 1 - r ** (0.5 + 0.5 * theta)
    alpha = torch.relu(1 - torch.abs(z_q)) ** 2
    rate = - torch.sum(
        (alpha * torch.log2(p0 * r ** torch.abs(z_q) + 1e-6)
        + (1 - alpha) * torch.log2(0.5 * (1 - p0) * (1 - r) * r ** (torch.abs(z_q) - 1) + 1e-6)),
        dim=-1
    )

    if reduce:
        rate = torch.mean(rate)

    return rate



def soft_dead_zone(x, dead_zone):
    """ approximates application of a dead zone to x """
    d = dead_zone * 0.05
    return x - d * torch.tanh(x / (0.1 + d))


def hard_quantize(x):
    """ round with copy gradient trick """
    return x + (torch.round(x) - x).detach()


def noise_quantize(x):
    """ simulates quantization with addition of random uniform noise """
    return x + (torch.rand_like(x) - 0.5)


# loss functions
class IDCT(nn.Module):
    def __init__(self, N, device=None):
        super(IDCT, self).__init__()

        self.N = N
        n = torch.arange(N, device=device)
        k = torch.arange(N, device=device)
        self.table = torch.cos(torch.pi/N * (n[:,None]+.5) * k[None,:])
        self.table[:,0] = self.table[:,0] * math.sqrt(.5)
        self.table = self.table / math.sqrt(N/2)

    def forward(self, x):
        return F.linear(x, self.table, None)

def dist_func(idct):
  def distortion_loss(y_true, y_pred, rate_lambda=None):
    """ custom distortion loss for LPCNet features """

    if y_true.size(-1) != 20:
        raise ValueError('distortion loss is designed to work with 20 features')

    ceps_error   = y_pred[..., :18] - y_true[..., :18]
    pitch_error  = 2*(y_pred[..., 18] - y_true[..., 18])
    corr_error   = y_pred[..., 19] - y_true[..., 19]
    pitch_weight = torch.relu(y_true[..., 19] + 0.5) ** 2
    band_pred = 10**(idct(y_pred[..., :18]))
    band_true = 10**(idct(y_true[..., :18]))
    band_pred = band_pred**.25
    band_true = band_true**.25

    #eband = (band_pred-band_true)**2 / (torch.std(band_true, axis=-2, keepdim=True)**2+.01*torch.mean(band_true**2, axis=-2, keepdim=True))
    eband = (band_pred-band_true)**2 / torch.mean(band_true**2, axis=-2, keepdim=True)
    eband = eband * (1.15-torch.arange(18)[None,None,:]/36).to(y_pred.device)
    eband = torch.mean(eband, axis=-1)
    eband4 = eband**2

    #loss = torch.mean(ceps_error ** 2 + (10/18) * torch.abs(pitch_error) * pitch_weight + (1/18) * corr_error ** 2, dim=-1)
    ceps_error = idct(ceps_error)
    ceps_error = torch.mean(ceps_error ** 2, dim=-1)

    loss = .05*ceps_error + 1*eband + torch.minimum(20*eband, 25*eband4) + 14/18 * torch.abs(pitch_error) * pitch_weight + 2/18 * corr_error ** 2


    if type(rate_lambda) != type(None):
        loss = loss / torch.sqrt(rate_lambda)

    loss = torch.mean(loss)

    return loss
  return distortion_loss

# sampling functions

import random


def random_split(start, stop, num_splits=3, min_len=3):
    get_min_len = lambda x : min([x[i+1] - x[i] for i in range(len(x) - 1)])
    candidate = [start] + sorted([random.randint(start, stop-1) for i in range(num_splits)]) + [stop]

    while get_min_len(candidate) < min_len:
        candidate = [start] + sorted([random.randint(start, stop-1) for i in range(num_splits)]) + [stop]

    return candidate



# weight initialization and clipping
def init_weights(module):

    if isinstance(module, nn.GRU):
        for p in module.named_parameters():
            if p[0].startswith('weight_hh_'):
                nn.init.orthogonal_(p[1])


def weight_clip_factory(max_value):
    """ weight clipping function concerning sum of abs values of adjacent weights """
    def clip_weight_(w):
        stop = w.size(1)
        # omit last column if stop is odd
        if stop % 2:
            stop -= 1
        max_values = max_value * torch.ones_like(w[:, :stop])
        factor = max_value / torch.maximum(max_values,
                                 torch.repeat_interleave(
                                     torch.abs(w[:, :stop:2]) + torch.abs(w[:, 1:stop:2]),
                                     2,
                                     1))
        with torch.no_grad():
            w[:, :stop] *= factor

    def clip_weights(module):
        if isinstance(module, nn.GRU) or isinstance(module, nn.Linear):
            for name, w in module.named_parameters():
                if name.startswith('weight'):
                    clip_weight_(w)

    return clip_weights

def n(x):
    return torch.clamp(x + (1./127.)*(torch.rand_like(x)-.5), min=-1., max=1.)

# RDOVAE module and submodules

sparsify_start     = 12000
sparsify_stop      = 24000
sparsify_interval  = 100
sparsify_exponent  = 3
#sparsify_start     = 0
#sparsify_stop      = 0

sparse_params1 = {
                'W_ir' : (0.6, [8, 4], False),
                'W_iz' : (0.4, [8, 4], False),
                'W_in' : (0.8, [8, 4], False)
                }

sparse_params2 = {
                'W_ir' : (0.35, [8, 4], False),
                'W_iz' : (0.25, [8, 4], False),
                'W_in' : (0.4, [8, 4], False)
                }

sparse_params3 = {
                'W_ir' : (0.25, [8, 4], False),
                'W_iz' : (0.15, [8, 4], False),
                'W_in' : (0.3, [8, 4], False)
                }

sparse_params4 = {
                'W_ir' : (0.15, [8, 4], False),
                'W_iz' : (0.1, [8, 4], False),
                'W_in' : (0.2, [8, 4], False)
                }

sparse_params5 = {
                'W_ir' : (0.15, [8, 4], False),
                'W_iz' : (0.1, [8, 4], False),
                'W_in' : (0.2, [8, 4], False)
                }

conv_params = [(1.0, [8, 4]),
               (0.8, [8, 4]),
               (0.6, [8, 4]),
               (0.4, [8, 4]),
               (0.3, [8, 4])]

dense_params = (0.5, [8, 4])

class MyConv(nn.Module):
    def __init__(self, input_dim, output_dim, dilation=1, softquant=False):
        super(MyConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation=dilation
        self.conv_dense = nn.Linear(input_dim, output_dim, bias=True)
        self.conv = nn.Conv1d(output_dim, output_dim, kernel_size=2, padding='valid', dilation=dilation)

        if softquant:
            self.conv = soft_quant(self.conv)

    def forward(self, x, state=None):
        device = x.device
        x = torch.tanh(self.conv_dense(x))
        conv_in = torch.cat([torch.zeros_like(x[:,0:self.dilation,:], device=device), x], -2).permute(0, 2, 1)
        return torch.tanh(self.conv(conv_in)).permute(0, 2, 1)

class GLU(nn.Module):
    def __init__(self, feat_size, softquant=False):
        super(GLU, self).__init__()

        torch.manual_seed(5)

        self.gate = weight_norm(nn.Linear(feat_size, feat_size, bias=False))

        if softquant:
            self.gate = soft_quant(self.gate)

        self.init_weights()

    def init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d)\
            or isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.orthogonal_(m.weight.data)

    def forward(self, x):

        out = x * torch.sigmoid(self.gate(x))

        return out

class CoreEncoder(nn.Module):
    STATE_HIDDEN = 128
    FRAMES_PER_STEP = 2
    CONV_KERNEL_SIZE = 4

    def __init__(self, feature_dim, output_dim, cond_size, cond_size2, state_size=24, softquant=False, adapt=False):
        """ core encoder for RDOVAE

            Computes latents, initial states, and rate estimates from features and lambda parameter

        """

        super(CoreEncoder, self).__init__()

        # hyper parameters
        self.feature_dim        = feature_dim
        self.output_dim         = output_dim
        self.cond_size          = cond_size
        self.cond_size2         = cond_size2
        self.state_size         = state_size

        # derived parameters
        self.input_dim = self.FRAMES_PER_STEP * self.feature_dim

        # layers
        self.dense_1 = nn.Linear(self.input_dim, 64)
        self.gru1 = nn.GRU(64, 32, batch_first=True)
        self.conv1 = MyConv(96, 64, softquant=True)
        self.gru2 = nn.GRU(160, 32, batch_first=True)
        self.conv2 = MyConv(192, 64, dilation=2, softquant=True)
        self.gru3 = nn.GRU(256, 32, batch_first=True)
        self.conv3 = MyConv(288, 64, dilation=2, softquant=True)
        self.gru4 = nn.GRU(352, 32, batch_first=True)
        self.conv4 = MyConv(384, 64, dilation=2, softquant=True)
        self.gru5 = nn.GRU(448, 32, batch_first=True)
        self.conv5 = MyConv(480, 64, dilation=2, softquant=True)

        self.z_dense = nn.Linear(544, self.output_dim)


        self.state_dense_1 = nn.Linear(544, self.STATE_HIDDEN)

        self.state_dense_2 = nn.Linear(self.STATE_HIDDEN, self.state_size)
        nb_params = sum(p.numel() for p in self.parameters())
        print(f"encoder: {nb_params} weights")
        # initialize weights
        if adapt:
            start = 0
            stop = 0
        else:
            start = sparsify_start
            stop = sparsify_stop
        self.apply(init_weights)
        self.sparsifier = []
        self.sparsifier.append(GRUSparsifier([(self.gru1, sparse_params1)], start, stop, sparsify_interval, sparsify_exponent))
        self.sparsifier.append(GRUSparsifier([(self.gru2, sparse_params2)], start, stop, sparsify_interval, sparsify_exponent))
        self.sparsifier.append(GRUSparsifier([(self.gru3, sparse_params3)], start, stop, sparsify_interval, sparsify_exponent))
        self.sparsifier.append(GRUSparsifier([(self.gru4, sparse_params4)], start, stop, sparsify_interval, sparsify_exponent))
        self.sparsifier.append(GRUSparsifier([(self.gru5, sparse_params5)], start, stop, sparsify_interval, sparsify_exponent))
        self.sparsifier.append(LinearSparsifier([(self.conv1.conv_dense, conv_params[1-1])], start, stop, sparsify_interval, sparsify_exponent))
        self.sparsifier.append(LinearSparsifier([(self.conv2.conv_dense, conv_params[2-1])], start, stop, sparsify_interval, sparsify_exponent))
        self.sparsifier.append(LinearSparsifier([(self.conv3.conv_dense, conv_params[3-1])], start, stop, sparsify_interval, sparsify_exponent))
        self.sparsifier.append(LinearSparsifier([(self.conv4.conv_dense, conv_params[4-1])], start, stop, sparsify_interval, sparsify_exponent))
        self.sparsifier.append(LinearSparsifier([(self.conv5.conv_dense, conv_params[5-1])], start, stop, sparsify_interval, sparsify_exponent))
        self.sparsifier.append(LinearSparsifier([(self.state_dense_1, dense_params)], start, stop, sparsify_interval, sparsify_exponent))

        # initialize weights
        self.apply(init_weights)

        if softquant:
            self.gru1 = soft_quant(self.gru1, names=['weight_hh_l0', 'weight_ih_l0'])
            self.gru2 = soft_quant(self.gru2, names=['weight_hh_l0', 'weight_ih_l0'])
            self.gru3 = soft_quant(self.gru3, names=['weight_hh_l0', 'weight_ih_l0'])
            self.gru4 = soft_quant(self.gru4, names=['weight_hh_l0', 'weight_ih_l0'])
            self.gru5 = soft_quant(self.gru5, names=['weight_hh_l0', 'weight_ih_l0'])
            self.z_dense = soft_quant(self.z_dense)
            self.state_dense_1 = soft_quant(self.state_dense_1)
            self.state_dense_2 = soft_quant(self.state_dense_2)


    def sparsify(self):
        for sparsifier in self.sparsifier:
            sparsifier.step()

    def forward(self, features):

        # reshape features
        x = torch.reshape(features, (features.size(0), features.size(1) // self.FRAMES_PER_STEP, self.FRAMES_PER_STEP * features.size(2)))

        batch = x.size(0)
        device = x.device

        # run encoding layer stack
        x = n(torch.tanh(self.dense_1(x)))
        x = torch.cat([x, n(self.gru1(x)[0])], -1)
        x = torch.cat([x, n(self.conv1(x))], -1)
        x = torch.cat([x, n(self.gru2(x)[0])], -1)
        x = torch.cat([x, n(self.conv2(x))], -1)
        x = torch.cat([x, n(self.gru3(x)[0])], -1)
        x = torch.cat([x, n(self.conv3(x))], -1)
        x = torch.cat([x, n(self.gru4(x)[0])], -1)
        x = torch.cat([x, n(self.conv4(x))], -1)
        x = torch.cat([x, n(self.gru5(x)[0])], -1)
        x = torch.cat([x, n(self.conv5(x))], -1)
        z = self.z_dense(x)

        # init state for decoder
        states = torch.tanh(self.state_dense_1(x))
        states = self.state_dense_2(states)

        return z, states




class CoreDecoder(nn.Module):

    FRAMES_PER_STEP = 4

    def __init__(self, input_dim, output_dim, cond_size, cond_size2, state_size=24, softquant=False, adapt=False):
        """ core decoder for RDOVAE

            Computes features from latents, initial state, and quantization index

        """

        super(CoreDecoder, self).__init__()

        # hyper parameters
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.cond_size  = cond_size
        self.cond_size2 = cond_size2
        self.state_size = state_size

        self.input_size = self.input_dim

        # layers
        self.dense_1    = nn.Linear(self.input_size, 96)
        self.gru1 = nn.GRU(96, 64, batch_first=True)
        self.conv1 = MyConv(160, 32, softquant=softquant)
        self.gru2 = nn.GRU(192, 64, batch_first=True)
        self.conv2 = MyConv(256, 32, softquant=softquant)
        self.gru3 = nn.GRU(288, 64, batch_first=True)
        self.conv3 = MyConv(352, 32, softquant=softquant)
        self.gru4 = nn.GRU(384, 64, batch_first=True)
        self.conv4 = MyConv(448, 32, softquant=softquant)
        self.gru5 = nn.GRU(480, 64, batch_first=True)
        self.conv5 = MyConv(544, 32, softquant=softquant)
        self.output  = nn.Linear(576, self.FRAMES_PER_STEP * self.output_dim)
        self.glu1 = GLU(64, softquant=softquant)
        self.glu2 = GLU(64, softquant=softquant)
        self.glu3 = GLU(64, softquant=softquant)
        self.glu4 = GLU(64, softquant=softquant)
        self.glu5 = GLU(64, softquant=softquant)
        self.hidden_init = nn.Linear(self.state_size, 128)
        self.gru_init = nn.Linear(128, 320)

        nb_params = sum(p.numel() for p in self.parameters())
        print(f"decoder: {nb_params} weights")
        # initialize weights
        if adapt:
            start = 0
            stop = 0
        else:
            start = sparsify_start
            stop = sparsify_stop
        self.apply(init_weights)
        self.sparsifier = []
        self.sparsifier.append(GRUSparsifier([(self.gru1, sparse_params1)], start, stop, sparsify_interval, sparsify_exponent))
        self.sparsifier.append(GRUSparsifier([(self.gru2, sparse_params2)], start, stop, sparsify_interval, sparsify_exponent))
        self.sparsifier.append(GRUSparsifier([(self.gru3, sparse_params3)], start, stop, sparsify_interval, sparsify_exponent))
        self.sparsifier.append(GRUSparsifier([(self.gru4, sparse_params4)], start, stop, sparsify_interval, sparsify_exponent))
        self.sparsifier.append(GRUSparsifier([(self.gru5, sparse_params5)], start, stop, sparsify_interval, sparsify_exponent))
        self.sparsifier.append(LinearSparsifier([(self.conv1.conv_dense, conv_params[1-1])], start, stop, sparsify_interval, sparsify_exponent))
        self.sparsifier.append(LinearSparsifier([(self.conv2.conv_dense, conv_params[2-1])], start, stop, sparsify_interval, sparsify_exponent))
        self.sparsifier.append(LinearSparsifier([(self.conv3.conv_dense, conv_params[3-1])], start, stop, sparsify_interval, sparsify_exponent))
        self.sparsifier.append(LinearSparsifier([(self.conv4.conv_dense, conv_params[4-1])], start, stop, sparsify_interval, sparsify_exponent))
        self.sparsifier.append(LinearSparsifier([(self.conv5.conv_dense, conv_params[5-1])], start, stop, sparsify_interval, sparsify_exponent))
        self.sparsifier.append(LinearSparsifier([(self.output, dense_params)], start, stop, sparsify_interval, sparsify_exponent))
        self.sparsifier.append(LinearSparsifier([(self.gru_init, dense_params)], start, stop, sparsify_interval, sparsify_exponent))

        if softquant:
            self.gru1 = soft_quant(self.gru1, names=['weight_hh_l0', 'weight_ih_l0'])
            self.gru2 = soft_quant(self.gru2, names=['weight_hh_l0', 'weight_ih_l0'])
            self.gru3 = soft_quant(self.gru3, names=['weight_hh_l0', 'weight_ih_l0'])
            self.gru4 = soft_quant(self.gru4, names=['weight_hh_l0', 'weight_ih_l0'])
            self.gru5 = soft_quant(self.gru5, names=['weight_hh_l0', 'weight_ih_l0'])
            self.output = soft_quant(self.output)
            self.gru_init = soft_quant(self.gru_init)

    def sparsify(self):
        for sparsifier in self.sparsifier:
            sparsifier.step()

    def forward(self, z, initial_state):

        hidden = torch.tanh(self.hidden_init(initial_state))
        gru_state = torch.tanh(self.gru_init(hidden).permute(1, 0, 2))
        h1_state = gru_state[:,:,:64].contiguous()
        h2_state = gru_state[:,:,64:128].contiguous()
        h3_state = gru_state[:,:,128:192].contiguous()
        h4_state = gru_state[:,:,192:256].contiguous()
        h5_state = gru_state[:,:,256:].contiguous()

        # run decoding layer stack
        x = n(torch.tanh(self.dense_1(z)))

        x = torch.cat([x, n(self.glu1(n(self.gru1(x, h1_state)[0])))], -1)
        x = torch.cat([x, n(self.conv1(x))], -1)
        x = torch.cat([x, n(self.glu2(n(self.gru2(x, h2_state)[0])))], -1)
        x = torch.cat([x, n(self.conv2(x))], -1)
        x = torch.cat([x, n(self.glu3(n(self.gru3(x, h3_state)[0])))], -1)
        x = torch.cat([x, n(self.conv3(x))], -1)
        x = torch.cat([x, n(self.glu4(n(self.gru4(x, h4_state)[0])))], -1)
        x = torch.cat([x, n(self.conv4(x))], -1)
        x = torch.cat([x, n(self.glu5(n(self.gru5(x, h5_state)[0])))], -1)
        x = torch.cat([x, n(self.conv5(x))], -1)

        # output layer and reshaping
        x10 = self.output(x)
        features = torch.reshape(x10, (x10.size(0), x10.size(1) * self.FRAMES_PER_STEP, x10.size(2) // self.FRAMES_PER_STEP))

        return features


class StatisticalModel(nn.Module):
    def __init__(self, quant_levels, latent_dim, state_dim):
        """ Statistical model for latent space

            Computes scaling, deadzone, r, and theta

        """

        super(StatisticalModel, self).__init__()

        # copy parameters
        self.latent_dim     = latent_dim
        self.state_dim      = state_dim
        self.total_dim      = latent_dim + state_dim
        self.quant_levels   = quant_levels
        self.embedding_dim  = 6 * self.total_dim

        # quantization embedding
        self.quant_embedding    = nn.Embedding(quant_levels, self.embedding_dim)

        # initialize embedding to 0
        with torch.no_grad():
            self.quant_embedding.weight[:] = 0


    def forward(self, quant_ids):
        """ takes quant_ids and returns statistical model parameters"""

        x = self.quant_embedding(quant_ids)

        # CAVE: theta_soft is not used anymore. Kick it out?
        quant_scale = F.softplus(x[..., 0 * self.total_dim : 1 * self.total_dim])
        dead_zone   = F.softplus(x[..., 1 * self.total_dim : 2 * self.total_dim])
        theta_soft  = torch.sigmoid(x[..., 2 * self.total_dim : 3 * self.total_dim])
        r_soft      = torch.sigmoid(x[..., 3 * self.total_dim : 4 * self.total_dim])
        theta_hard  = torch.sigmoid(x[..., 4 * self.total_dim : 5 * self.total_dim])
        r_hard      = torch.sigmoid(x[..., 5 * self.total_dim : 6 * self.total_dim])


        return {
            'quant_embedding'   : x,
            'quant_scale'       : quant_scale,
            'dead_zone'         : dead_zone,
            'r_hard'            : r_hard,
            'theta_hard'        : theta_hard,
            'r_soft'            : r_soft,
            'theta_soft'        : theta_soft
        }


class RDOVAE(nn.Module):
    def __init__(self,
                 feature_dim,
                 latent_dim,
                 quant_levels,
                 cond_size,
                 cond_size2,
                 state_dim=24,
                 split_mode='split',
                 chunks_per_offset=4,
                 clip_weights=False,
                 pvq_num_pulses=82,
                 state_dropout_rate=0,
                 softquant=False,
                 adapt=False):

        super(RDOVAE, self).__init__()

        self.feature_dim    = feature_dim
        self.latent_dim     = latent_dim
        self.quant_levels   = quant_levels
        self.cond_size      = cond_size
        self.cond_size2     = cond_size2
        self.split_mode     = split_mode
        self.chunks_per_offset = chunks_per_offset
        self.state_dim      = state_dim
        self.pvq_num_pulses = pvq_num_pulses
        self.state_dropout_rate = state_dropout_rate

        # submodules encoder and decoder share the statistical model
        self.statistical_model = StatisticalModel(quant_levels, latent_dim, state_dim)
        self.core_encoder = nn.DataParallel(CoreEncoder(feature_dim, latent_dim, cond_size, cond_size2, state_size=state_dim, softquant=softquant, adapt=adapt))
        self.core_decoder = nn.DataParallel(CoreDecoder(latent_dim+1, feature_dim, cond_size, cond_size2, state_size=state_dim, softquant=softquant, adapt=adapt))

        self.enc_stride = CoreEncoder.FRAMES_PER_STEP
        self.dec_stride = CoreDecoder.FRAMES_PER_STEP

        if clip_weights:
            self.weight_clip_fn = weight_clip_factory(0.496)
        else:
            self.weight_clip_fn = None

        if self.dec_stride % self.enc_stride != 0:
            raise ValueError(f"get_decoder_chunks_generic: encoder stride does not divide decoder stride")

    def clip_weights(self):
        if not type(self.weight_clip_fn) == type(None):
            self.apply(self.weight_clip_fn)

    def sparsify(self):
        self.core_encoder.module.sparsify()
        self.core_decoder.module.sparsify()

    def get_decoder_chunks(self, z_frames, mode='split', chunks_per_offset = 4):

        enc_stride = self.enc_stride
        dec_stride = self.dec_stride

        stride = dec_stride // enc_stride

        chunks = []

        for offset in range(stride):
            # start is the smallest number = offset mod stride that decodes to a valid range
            start = offset
            while enc_stride * (start + 1) - dec_stride < 0:
                start += stride

            # check if start is a valid index
            if start >= z_frames:
                raise ValueError("get_decoder_chunks_generic: range too small")

            # stop is the smallest number outside [0, num_enc_frames] that's congruent to offset mod stride
            stop = z_frames - (z_frames % stride) + offset
            while stop < z_frames:
                stop += stride

            # calculate split points
            length = (stop - start)
            if mode == 'split':
                split_points = [start + stride * int(i * length / chunks_per_offset / stride) for i in range(chunks_per_offset)] + [stop]
            elif mode == 'random_split':
                split_points = [stride * x + start for x in random_split(0, (stop - start)//stride - 1, chunks_per_offset - 1, 1)]
            elif mode == 'skewed_split':
                split_points = [start + stride * int(i * length / 4 / chunks_per_offset / stride) for i in range(chunks_per_offset)] + [stop]
            else:
                raise ValueError(f"get_decoder_chunks_generic: unknown mode {mode}")


            for i in range(chunks_per_offset):
                # (enc_frame_start, enc_frame_stop, enc_frame_stride, stride, feature_frame_start, feature_frame_stop)
                # encoder range(i, j, stride) maps to feature range(enc_stride * (i + 1) - dec_stride, enc_stride * j)
                # provided that i - j = 1 mod stride
                chunks.append({
                    'z_start'         : split_points[i],
                    'z_stop'          : split_points[i + 1] - stride + 1,
                    'z_stride'        : stride,
                    'features_start'  : enc_stride * (split_points[i] + 1) - dec_stride,
                    'features_stop'   : enc_stride * (split_points[i + 1] - stride + 1)
                })

        return chunks


    def forward(self, features, q_id):

        # calculate statistical model from quantization ID
        statistical_model = self.statistical_model(q_id)

        # run encoder
        z, states = self.core_encoder(features)

        # scaling, dead-zone and quantization
        z = z * statistical_model['quant_scale'][:,:,:self.latent_dim]
        z = soft_dead_zone(z, statistical_model['dead_zone'][:,:,:self.latent_dim])

        # quantization
        z_q = hard_quantize(z) / statistical_model['quant_scale'][:,:,:self.latent_dim]
        z_n = noise_quantize(z) / statistical_model['quant_scale'][:,:,:self.latent_dim]
        z_q = torch.cat([z_q, q_id[:,:,None]*.125-1], -1)
        z_n = torch.cat([z_n, q_id[:,:,None]*.125-1], -1)
        #states_q = soft_pvq(states, self.pvq_num_pulses)
        states = states * statistical_model['quant_scale'][:,:,self.latent_dim:]
        states = soft_dead_zone(states, statistical_model['dead_zone'][:,:,self.latent_dim:])

        states_q = hard_quantize(states) / statistical_model['quant_scale'][:,:,self.latent_dim:]
        states_n = noise_quantize(states) / statistical_model['quant_scale'][:,:,self.latent_dim:]

        if self.state_dropout_rate > 0:
            drop = torch.rand(states_q.size(0)) < self.state_dropout_rate
            mask = torch.ones_like(states_q)
            mask[drop] = 0
            states_q = states_q * mask

        # decoder
        chunks = self.get_decoder_chunks(z.size(1), mode=self.split_mode, chunks_per_offset=self.chunks_per_offset)

        outputs_hq = []
        outputs_sq = []
        for chunk in chunks:
            # decoder with hard quantized input
            z_dec_reverse       = torch.flip(z_q[..., chunk['z_start'] : chunk['z_stop'] : chunk['z_stride'], :], [1])
            dec_initial_state   = states_q[..., chunk['z_stop'] - 1 : chunk['z_stop'], :]
            features_reverse = self.core_decoder(z_dec_reverse,  dec_initial_state)
            outputs_hq.append((torch.flip(features_reverse, [1]), chunk['features_start'], chunk['features_stop']))


            # decoder with soft quantized input
            z_dec_reverse       = torch.flip(z_n[..., chunk['z_start'] : chunk['z_stop'] : chunk['z_stride'], :],  [1])
            dec_initial_state   = states_n[..., chunk['z_stop'] - 1 : chunk['z_stop'], :]
            features_reverse    = self.core_decoder(z_dec_reverse, dec_initial_state)
            outputs_sq.append((torch.flip(features_reverse, [1]), chunk['features_start'], chunk['features_stop']))

        return {
            'outputs_hard_quant' : outputs_hq,
            'outputs_soft_quant' : outputs_sq,
            'z'                 : z,
            'states'            : states,
            'statistical_model' : statistical_model
        }

    def encode(self, features):
        """ encoder with quantization and rate estimation """

        z, states = self.core_encoder(features)

        # quantization of initial states
        states = soft_pvq(states, self.pvq_num_pulses)
        state_size = m.log2(pvq_codebook_size(self.state_dim, self.pvq_num_pulses))

        return z, states, state_size

    def decode(self, z, initial_state):
        """ decoder (flips sequences by itself) """

        z_reverse       = torch.flip(z, [1])
        features_reverse = self.core_decoder(z_reverse, initial_state)
        features = torch.flip(features_reverse, [1])

        return features

    def quantize(self, z, q_ids):
        """ quantization of latent vectors """

        stats = self.statistical_model(q_ids)

        zq = z * stats['quant_scale'][:self.latent_dim]
        zq = soft_dead_zone(zq, stats['dead_zone'][:self.latent_dim])
        zq = torch.round(zq)

        sizes = hard_rate_estimate(zq, stats['r_hard'][:,:,:self.latent_dim], stats['theta_hard'][:,:,:self.latent_dim], reduce=False)

        return zq, sizes

    def unquantize(self, zq, q_ids):
        """ re-scaling of latent vector """

        stats = self.statistical_model(q_ids)

        z = zq / stats['quant_scale'][:,:,:self.latent_dim]

        return z

    def freeze_model(self):

        # freeze all parameters
        for p in self.parameters():
            p.requires_grad = False

        for p in self.statistical_model.parameters():
            p.requires_grad = True
