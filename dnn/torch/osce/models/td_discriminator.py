"""
MIT License

Copyright (c) 2020 Jungil Kong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# This is an adaptation of the HiFi-Gan discriminators derived from https://github.com/jik876/hifi-gan

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

LRELU_SLOPE = 0.1

class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False, max_channels=1024):
        super(DiscriminatorP, self).__init__()
        self.max_channels = max_channels
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(min(self.max_channels, 128), min(self.max_channels, 512), (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(min(self.max_channels, 512), min(self.max_channels, 1024), (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(min(self.max_channels, 1024), min(self.max_channels, 1024), (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(min(self.max_channels, 1024), 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        output = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            output.append(x)
        x = self.conv_post(x)
        output.append(x)

        return output


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, max_channels=1024):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2, max_channels=max_channels),
            DiscriminatorP(3, max_channels=max_channels),
            DiscriminatorP(5, max_channels=max_channels),
            DiscriminatorP(7, max_channels=max_channels),
            DiscriminatorP(11, max_channels=max_channels),
        ])

    def forward(self, y):
        outputs = []
        for disc in self.discriminators:
            outputs.append(disc(y))

        return outputs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False, max_channels=1024):
        super(DiscriminatorS, self).__init__()
        self.max_channels = max_channels
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, min(self.max_channels, 128), 15, 1, padding=7)),
            norm_f(Conv1d(min(self.max_channels, 128), min(self.max_channels, 128), 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(min(self.max_channels, 128), min(self.max_channels, 256), 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(min(self.max_channels, 256), min(self.max_channels, 512), 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(min(self.max_channels, 512), min(self.max_channels, 1024), 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(min(self.max_channels, 1024), min(self.max_channels, 1024), 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(min(self.max_channels, 1024), min(self.max_channels, 1024), 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(min(self.max_channels, 1024), 1, 3, 1, padding=1))

    def forward(self, x):
        output = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            output.append(x)
        x = self.conv_post(x)
        output.append(x)

        return output


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self, max_channels=1024):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True, max_channels=max_channels),
            DiscriminatorS(max_channels=max_channels),
            DiscriminatorS(max_channels=max_channels),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y):
        outputs = []
        for disc in self.discriminators:
            outputs.append(disc(y))

        return outputs


class TDMultiResolutionDiscriminator(torch.nn.Module):
    def __init__(self, max_channels=1024, **kwargs):
        super().__init__()
        print(f"{max_channels=}")
        self.msd = MultiScaleDiscriminator(max_channels=max_channels)
        self.mpd = MultiPeriodDiscriminator(max_channels=max_channels)

    def forward(self, y):
        return self.msd(y) + self.mpd(y)