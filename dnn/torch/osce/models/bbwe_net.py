import torch
from torch import nn
import torch.nn.functional as F

from utils.complexity import _conv1d_flop_count
from utils.layers.silk_upsampler import SilkUpsampler
from utils.layers.limited_adaptive_conv1d import LimitedAdaptiveConv1d
from utils.layers.td_shaper import TDShaper
from dnntools.quantization.softquant import soft_quant

DUMP=False

if DUMP:
    from scipy.io import wavfile
    import numpy as np
    import os

    os.makedirs('dump', exist_ok=True)

    def dump_as_wav(filename, fs, x):
        s  = x.cpu().squeeze().flatten().numpy()
        s = 0.5 * s / s.max()
        wavfile.write(filename, fs, (2**15 * s).astype(np.int16))

DEBUGDUMP=False
if DEBUGDUMP:
    import os
    debugdumpdir='debugdump'
    os.makedirs(debugdumpdir, exist_ok=True)

    def debugdump(filename, data):
        data.detach().numpy().tofile(os.path.join(debugdumpdir, filename))


class FloatFeatureNet(nn.Module):

    def __init__(self,
                 feature_dim=84,
                 num_channels=256,
                 upsamp_factor=2,
                 lookahead=False,
                 kernel_size1=3,
                 softquant=False):

        super().__init__()

        self.feature_dim = feature_dim
        self.num_channels = num_channels
        self.upsamp_factor = upsamp_factor
        self.lookahead = lookahead
        self.kernel_size1 = kernel_size1

        assert self.kernel_size1 % 2 == 1

        self.conv1 = nn.Conv1d(feature_dim, num_channels, self.kernel_size1)
        self.conv2 = nn.Conv1d(num_channels, num_channels, 3)

        self.gru = nn.GRU(num_channels, num_channels, batch_first=True)

        self.tconv = nn.ConvTranspose1d(num_channels, num_channels, upsamp_factor, upsamp_factor)

        if softquant:
            self.conv2 = soft_quant(self.conv2)
            self.gru   = soft_quant(self.gru, names=['weight_hh_l0', 'weight_ih_l0'])
            self.tconv = soft_quant(self.tconv)

    def flop_count(self, rate=100):
        count = 0
        for conv in self.conv1, self.conv2, self.tconv:
            count += _conv1d_flop_count(conv, rate)

        count += 2 * (3 * self.gru.input_size * self.gru.hidden_size + 3 * self.gru.hidden_size * self.gru.hidden_size) * self.upsamp_factor * rate

        return count


    def forward(self, features, state=None, debug=False):
        """ features shape: (batch_size, num_frames, feature_dim) """

        batch_size = features.size(0)

        if DEBUGDUMP:
            debugdump('features.f32', features.float())

        if state is None:
            state = torch.zeros((1, batch_size, self.num_channels), device=features.device)


        features = features.permute(0, 2, 1)
        if self.lookahead:
            c = torch.tanh(self.conv1(F.pad(features, [self.kernel_size1 // 2, self.kernel_size1//2])))
            c = torch.tanh(self.conv2(F.pad(c, [2, 0])))
        else:
            c = torch.tanh(self.conv1(F.pad(features, [self.kernel_size1 - 1, 0])))
            if DEBUGDUMP:
                debugdump('feature_net_conv1_activated.f32', c.permute(0, 2, 1))
            c = torch.tanh(self.conv2(F.pad(c, [2, 0])))
            if DEBUGDUMP:
                debugdump('feature_net_conv2_activated.f32', c.permute(0, 2, 1))

        c = torch.tanh(self.tconv(c))
        if DEBUGDUMP:
            debugdump('feature_net_tconv_activated.f32', c.permute(0, 2, 1))

        c = c.permute(0, 2, 1)

        c, _ = self.gru(c, state)

        if DEBUGDUMP:
            debugdump('feature_net_gru.f32', c)

        return c


class Folder(torch.nn.Module):
    def __init__(self, num_taps, frame_size):
        super().__init__()

        self.num_taps = num_taps
        self.frame_size = frame_size
        assert frame_size % num_taps == 0
        self.taps = torch.nn.Parameter(torch.randn(num_taps).view(1, 1, -1), requires_grad=True)


    def flop_count(self, rate):

        # single multiplication per sample
        return rate

    def forward(self, x, *args):

        batch_size, num_channels, length = x.shape
        assert length % self.num_taps == 0

        y = x * torch.repeat_interleave(torch.exp(self.taps), length // self.num_taps, dim=-1)

        return y

class BBWENet(torch.nn.Module):
    FRAME_SIZE16k=80

    def __init__(self,
                 feature_dim,
                 cond_dim=128,
                 kernel_size1=3,
                 kernel_size16=15,
                 kernel_size32=15,
                 kernel_size48=15,
                 conv_gain_limits_db=[-12, 12], # might be a bit tight
                 activation="ImPowI",
                 avg_pool_k32 = 8,
                 avg_pool_k48 = 12,
                 interpolate_k32=1,
                 interpolate_k48=1,
                 shape_extension=True,
                 func_extension=True,
                 shaper='TDShaper',
                 bias=False,
                 softquant=False,
                 lookahead=False,
                 ):

        super().__init__()


        self.feature_dim            = feature_dim
        self.cond_dim               = cond_dim
        self.kernel_size1           = kernel_size1
        self.kernel_size16          = kernel_size16
        self.kernel_size32          = kernel_size32
        self.kernel_size48          = kernel_size48
        self.conv_gain_limits_db    = conv_gain_limits_db
        self.activation             = activation
        self.shape_extension        = shape_extension
        self.func_extension         = func_extension
        self.shaper                 = shaper

        assert (shape_extension or func_extension) and "Require at least one of shape_extension and func_extension to be true"


        self.frame_size16 = 1 * self.FRAME_SIZE16k
        self.frame_size32 = 2 * self.FRAME_SIZE16k
        self.frame_size48 = 3 * self.FRAME_SIZE16k

        # upsampler
        self.upsampler = SilkUpsampler()

        # feature net
        self.feature_net = FloatFeatureNet(feature_dim=feature_dim, num_channels=cond_dim, kernel_size1=kernel_size1, softquant=softquant, lookahead=lookahead)

        # non-linear transforms

        if self.shape_extension:
            if self.shaper == 'TDShaper':
                self.tdshape1 = TDShaper(cond_dim, frame_size=self.frame_size32, avg_pool_k=avg_pool_k32, interpolate_k=interpolate_k32, bias=bias, softquant=softquant)
                self.tdshape2 = TDShaper(cond_dim, frame_size=self.frame_size48, avg_pool_k=avg_pool_k48, interpolate_k=interpolate_k48, bias=bias, softquant=softquant)
            elif self.shaper == 'Folder':
                self.tdshape1 = Folder(8, frame_size=self.frame_size32)
                self.tdshape2 = Folder(12, frame_size=self.frame_size48)
            else:
                raise ValueError(f"unknown shaper {self.shaper}")

        if activation == 'ImPowI':
            self.nlfunc = lambda x : x * torch.sin(torch.log(torch.abs(x) + 1e-6))
        elif activation == "ReLU":
            self.nlfunc = F.relu
        else:
            raise ValueError(f"unknown activation {activation}")

        latent_channels = 1
        if self.shape_extension: latent_channels += 1
        if self.func_extension: latent_channels += 1

        # spectral shaping
        self.af1 = LimitedAdaptiveConv1d(1, latent_channels, self.kernel_size16, cond_dim, frame_size=self.frame_size16, overlap_size=self.frame_size16//2, use_bias=False, padding=[self.kernel_size16 - 1, 0], gain_limits_db=conv_gain_limits_db, norm_p=2, softquant=softquant)
        self.af2 = LimitedAdaptiveConv1d(latent_channels, latent_channels, self.kernel_size32, cond_dim, frame_size=self.frame_size32, overlap_size=self.frame_size32//2, use_bias=False, padding=[self.kernel_size32 - 1, 0], gain_limits_db=conv_gain_limits_db, norm_p=2, softquant=softquant)
        self.af3 = LimitedAdaptiveConv1d(latent_channels, 1, self.kernel_size48, cond_dim, frame_size=self.frame_size48, overlap_size=self.frame_size48//2, use_bias=False, padding=[self.kernel_size48 - 1, 0], gain_limits_db=conv_gain_limits_db, norm_p=2, softquant=softquant)


    def flop_count(self, rate=16000, verbose=False):

        frame_rate = rate / self.FRAME_SIZE16k

        # feature net
        feature_net_flops = self.feature_net.flop_count(frame_rate // 2)
        af_flops = self.af1.flop_count(rate) + self.af2.flop_count(2 * rate) + self.af3.flop_count(3 * rate)

        if self.shape_extension:
            shape_flops = self.tdshape1.flop_count(2*rate) + self.tdshape2.flop_count(3*rate)
        else:
            shape_flops = 0

        if verbose:
            print(f"feature net: {feature_net_flops / 1e6} MFLOPS")
            print(f"shape flops: {shape_flops / 1e6} MFLOPS")
            print(f"adaptive conv: {af_flops / 1e6} MFLOPS")

        return feature_net_flops + af_flops + shape_flops

    def forward(self, x, features, debug=False):

        cf = self.feature_net(features)

        # split into latent_channels channels
        y16 = self.af1(x, cf, debug=debug)

        if DEBUGDUMP:
            debugdump('bbwenet_af1_1.f32', y16[:, 0:1, :]) # first channel is bypass channel
            debugdump('bbwenet_af1_2.f32', y16[:, 1:2, :])
            debugdump('bbwenet_af1_3.f32', y16[:, 2:3, :])

        # first 2x upsampling step
        y32 = self.upsampler.hq_2x_up(y16)
        y32_out = y32[:, 0:1, :] # first channel is bypass channel

        if DEBUGDUMP:
            debugdump('bbwenet_up2_1.f32', y32_out)
            debugdump('bbwenet_up2_2.f32', y32[:, 1:2, :])
            debugdump('bbwenet_up2_3.f32', y32[:, 2:3, :])

        # extend frequencies
        idx = 1
        if self.shape_extension:
            y32_shape = self.tdshape1(y32[:, idx:idx+1, :], cf)
            y32_out = torch.cat((y32_out, y32_shape), dim=1)
            if DEBUGDUMP:
                debugdump('bbwenet_up2_shape.f32', y32_shape)
            idx += 1

        if self.func_extension:
            y32_func = self.nlfunc(y32[:, idx:idx+1, :])
            y32_out = torch.cat((y32_out, y32_func), dim=1)
            if DEBUGDUMP:
                debugdump('bbwenet_up2_func.f32', y32_func)

        # mix-select
        y32_out = self.af2(y32_out, cf)
        if DEBUGDUMP:
            debugdump('bbwenet_af2_1.f32', y32_out[:, 0:1, :])
            debugdump('bbwenet_af2_2.f32', y32_out[:, 1:2, :])
            debugdump('bbwenet_af2_3.f32', y32_out[:, 2:3, :])

        # 1.5x upsampling
        y48 = self.upsampler.interpolate_3_2(y32_out)
        y48_out = y48[:, 0:1, :] # first channel is bypass channel

        if DEBUGDUMP:
            debugdump('bbwenet_up15_1.f32', y48_out)
            debugdump('bbwenet_up15_2.f32', y48[:, 1:2, :])
            debugdump('bbwenet_up15_3.f32', y48[:, 2:3, :])

        # extend frequencies
        idx = 1
        if self.shape_extension:
            y48_shape = self.tdshape2(y48[:, idx:idx+1, :], cf)
            y48_out = torch.cat((y48_out, y48_shape), dim=1)
            idx += 1
            if DEBUGDUMP:
                debugdump('bbwenet_up15_shape.f32', y48_shape)

        if self.func_extension:
            y48_func = self.nlfunc(y48[:, idx:idx+1, :])
            y48_out = torch.cat((y48_out, y48_func), dim=1)
            if DEBUGDUMP:
                debugdump('bbwenet_up15_func.f32', y48_func)

        # 2nd mixing
        y48_out = self.af3(y48_out, cf)

        if DEBUGDUMP:
            debugdump('bbwenet_af3_1.f32', y48_out)

        return y48_out
