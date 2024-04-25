import torch
from torch import nn
import torch.nn.functional as F

from utils.complexity import _conv1d_flop_count
from utils.layers.silk_upsampler import SilkUpsampler
from utils.layers.limited_adaptive_conv1d import LimitedAdaptiveConv1d
from utils.layers.td_shaper import TDShaper

class FloatFeatureNet(nn.Module):

    def __init__(self,
                 feature_dim=84,
                 num_channels=256,
                 upsamp_factor=2,
                 lookahead=False):

        super().__init__()

        self.feature_dim = feature_dim
        self.num_channels = num_channels
        self.upsamp_factor = upsamp_factor
        self.lookahead = lookahead

        self.conv1 = nn.Conv1d(feature_dim, num_channels, 3)
        self.conv2 = nn.Conv1d(num_channels, num_channels, 3)

        self.gru = nn.GRU(num_channels, num_channels, batch_first=True)

        self.tconv = nn.ConvTranspose1d(num_channels, num_channels, upsamp_factor, upsamp_factor)

    def flop_count(self, rate=100):
        count = 0
        for conv in self.conv1, self.conv2, self.tconv:
            count += _conv1d_flop_count(conv, rate)

        count += 2 * (3 * self.gru.input_size * self.gru.hidden_size + 3 * self.gru.hidden_size * self.gru.hidden_size) * rate

        return count


    def forward(self, features, state=None):
        """ features shape: (batch_size, num_frames, feature_dim) """

        batch_size = features.size(0)

        if state is None:
            state = torch.zeros((1, batch_size, self.num_channels), device=features.device)


        features = features.permute(0, 2, 1)
        if self.lookahead:
            c = torch.tanh(self.conv1(F.pad(features, [1, 1])))
            c = torch.tanh(self.conv2(F.pad(c, [2, 0])))
        else:
            c = torch.tanh(self.conv1(F.pad(features, [2, 0])))
            c = torch.tanh(self.conv2(F.pad(c, [2, 0])))

        c = torch.tanh(self.tconv(c))

        c = c.permute(0, 2, 1)

        c, _ = self.gru(c, state)

        return c



class BWENet(torch.nn.Module):
    FRAME_SIZE16k=80

    def __init__(self,
                 feature_dim,
                 cond_dim=128,
                 kernel_size32=15,
                 kernel_size48=15,
                 conv_gain_limits_db=[-12, 12],
                 activation="AdaShape"
                 ):

        super().__init__()


        self.feature_dim            = feature_dim
        self.cond_dim               = cond_dim
        self.kernel_size32          = kernel_size32
        self.kernel_size48          = kernel_size48
        self.conv_gain_limits_db    = conv_gain_limits_db
        self.activation             = activation

        self.frame_size32 = 2 * self.FRAME_SIZE16k
        self.frame_size48 = 3 * self.FRAME_SIZE16k

        # upsampler
        self.upsampler = SilkUpsampler()

        # feature net
        self.feature_net = FloatFeatureNet(feature_dim=feature_dim, num_channels=cond_dim)

        # non-linear transforms
        if activation == "AdaShape":
            self.tdshape1 = TDShaper(cond_dim, frame_size=self.frame_size32, avg_pool_k=8)
            self.tdshape2 = TDShaper(cond_dim, frame_size=self.frame_size48, avg_pool_k=12)
            self.act1 = self.tdshape1
            self.act2 = self.tdshape2
        elif activation == "ReLU":
            self.act1 = lambda x, _: F.relu(x)
            self.act2 = lambda x, _: F.relu(x)
        else:
            raise ValueError(f"unknown activation {activation}")

        # spectral shaping
        self.af1 = LimitedAdaptiveConv1d(1, 2, self.kernel_size32, cond_dim, frame_size=self.frame_size32, overlap_size=self.frame_size32//2, use_bias=False, padding=[self.kernel_size32 - 1, 0], gain_limits_db=conv_gain_limits_db, norm_p=2)
        self.af2 = LimitedAdaptiveConv1d(2, 1, self.kernel_size32, cond_dim, frame_size=self.frame_size32, overlap_size=self.frame_size32//2, use_bias=False, padding=[self.kernel_size32 - 1, 0], gain_limits_db=conv_gain_limits_db, norm_p=2)
        self.af3 = LimitedAdaptiveConv1d(1, 2, self.kernel_size48, cond_dim, frame_size=self.frame_size48, overlap_size=self.frame_size48//2, use_bias=False, padding=[self.kernel_size48 - 1, 0], gain_limits_db=conv_gain_limits_db, norm_p=2)
        self.af4 = LimitedAdaptiveConv1d(2, 1, self.kernel_size48, cond_dim, frame_size=self.frame_size48, overlap_size=self.frame_size48//2, use_bias=False, padding=[self.kernel_size48 - 1, 0], gain_limits_db=conv_gain_limits_db, norm_p=2)


    def flop_count(self, rate=16000, verbose=False):

        frame_rate = rate / self.FRAME_SIZE16k

        # feature net
        feature_net_flops = self.feature_net.flop_count(frame_rate)
        af_flops = self.af1.flop_count(rate) + self.af2.flop_count(2 * rate) + self.af3.flop_count(3 * rate) + + self.af4.flop_count(3 * rate)

        if verbose:
            print(f"feature net: {feature_net_flops / 1e6} MFLOPS")
            print(f"adaptive conv: {af_flops / 1e6} MFLOPS")

        return feature_net_flops + af_flops

    def forward(self, x, features, debug=False):

        cf = self.feature_net(features)

        # first 2x upsampling step
        y32 = self.upsampler.hq_2x_up(x)

        # split
        y32 = self.af1(y32, cf, debug=debug)

        # activation
        y32_1 = y32[:, 0:1, :]
        y32_2 = self.act1(y32[:, 1:2, :], cf)

        # mix
        y32 = torch.cat((y32_1, y32_2), dim=1)
        y32 = self.af2(y32, cf, debug=debug)

        # 1.5x interpolation
        y48 = self.upsampler.interpolate_3_2(y32)

        # split
        y48 = self.af3(y48, cf, debug=debug)

        # activate
        y48_1 = y48[:, 0:1, :]
        y48_2 = self.act2(y48[:, 1:2, :], cf)

        # mix
        y48 = torch.cat((y48_1, y48_2), dim=1)
        y48 = self.af4(y48, cf, debug=debug)

        return y48