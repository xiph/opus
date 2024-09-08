import torch
from torch import nn
import torch.nn.functional as F
import scipy.signal

from utils.complexity import _conv1d_flop_count
from utils.softquant import soft_quant

class TDShaper(nn.Module):
    COUNTER = 1

    def __init__(self,
                 feature_dim,
                 frame_size=160,
                 innovate=False,
                 avg_pool_k=4,
                 pool_after=False,
                 softquant=False,
                 apply_weight_norm=False,
                 interpolate_k=1,
                 noise_substitution=False,
                 cutoff=None,
                 bias=True,
    ):
        """

        Parameters:
        -----------


        feature_dim : int
            dimension of input features

        frame_size : int
            frame size

        avg_pool_k : int, optional
            kernel size and stride for avg pooling

        padding : List[int, int]

        """

        super().__init__()

        if innovate:
            print("warning: option innovate is no longer supported, setting innovate to True will have no effect")

        self.feature_dim    = feature_dim
        self.frame_size     = frame_size
        self.avg_pool_k     = avg_pool_k
        self.pool_after     = pool_after
        self.interpolate_k  = interpolate_k
        self.hidden_dim     = frame_size // interpolate_k
        self.innovate       = innovate
        self.noise_substitution = noise_substitution
        self.cutoff             = cutoff

        assert frame_size % avg_pool_k == 0
        assert frame_size % interpolate_k == 0
        self.env_dim = frame_size // avg_pool_k + 1

        norm = torch.nn.utils.weight_norm if apply_weight_norm else lambda x, name=None: x

        # feature transform
        self.feature_alpha1_f = norm(nn.Conv1d(self.feature_dim, self.hidden_dim, 2, bias=bias))
        self.feature_alpha1_t = norm(nn.Conv1d(self.env_dim, self.hidden_dim, 2, bias=bias))
        self.feature_alpha2 = norm(nn.Conv1d(self.hidden_dim, self.hidden_dim, 2, bias=bias))

        self.interpolate_weight = nn.Parameter(torch.ones(1, 1, self.interpolate_k) / self.interpolate_k, requires_grad=False)

        if softquant:
            self.feature_alpha1_f = soft_quant(self.feature_alpha1_f)

        if self.noise_substitution:
            self.hp = torch.nn.Parameter(torch.from_numpy(scipy.signal.firwin(15, cutoff, pass_zero=False)).float().view(1, 1, -1), requires_grad=False)
        else:
            self.hp = None


    def flop_count(self, rate):

        frame_rate = rate / self.frame_size

        shape_flops = sum([_conv1d_flop_count(x, frame_rate) for x in (self.feature_alpha1_f, self.feature_alpha1_t, self.feature_alpha2)]) + 11 * frame_rate * self.frame_size

        return shape_flops

    def envelope_transform(self, x):

        x = torch.abs(x)
        if self.pool_after:
            x = torch.log(x + .5**16)
            x = F.avg_pool1d(x, self.avg_pool_k, self.avg_pool_k)
        else:
            x = F.avg_pool1d(x, self.avg_pool_k, self.avg_pool_k)
            x = torch.log(x + .5**16)

        x = x.reshape(x.size(0), -1, self.env_dim - 1)
        avg_x = torch.mean(x, -1, keepdim=True)

        x = torch.cat((x - avg_x, avg_x), dim=-1)

        return x

    def forward(self, x, features, debug=False):
        """ innovate signal parts with temporal shaping


        Parameters:
        -----------
        x : torch.tensor
            input signal of shape (batch_size, 1, num_samples)

        features : torch.tensor
            frame-wise features of shape (batch_size, num_frames, feature_dim)

        """

        batch_size = x.size(0)
        num_samples = x.size(2)

        # generate temporal envelope
        tenv = self.envelope_transform(x)

        # feature path
        f = F.pad(features.permute(0, 2, 1), [1, 0])
        t = F.pad(tenv.permute(0, 2, 1), [1, 0])
        alpha = self.feature_alpha1_f(f) + self.feature_alpha1_t(t)
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = self.feature_alpha2(F.pad(alpha, [1, 0]))
        # reshape and interpolate to size (batch_size, 1, num_samples)
        alpha = alpha.permute(0, 2, 1)
        alpha = alpha.reshape(batch_size, 1, num_samples // self.interpolate_k)
        if self.interpolate_k != 1:
            alpha = F.interpolate(alpha, self.interpolate_k * alpha.size(-1), mode='nearest')
            alpha = F.conv1d(F.pad(alpha, [self.interpolate_k - 1, 0], mode='reflect'), self.interpolate_weight) # interpolation in log-domain
        alpha = torch.exp(alpha)

        # sample-wise shaping in time domain
        if self.noise_substitution:
            if self.hp is not None:
                x = torch.rand_like(x)
                x = F.pad(x, [7, 7], mode='reflect')
                x = F.conv1d(x, self.hp)
            else:
                x = 2 * torch.rand_like(x) - 1

        y = alpha * x

        return y
