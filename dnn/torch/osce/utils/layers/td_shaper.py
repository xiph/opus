import torch
from torch import nn
import torch.nn.functional as F

from utils.complexity import _conv1d_flop_count

class TDShaper(nn.Module):
    COUNTER = 1

    def __init__(self,
                 feature_dim,
                 frame_size=160,
                 avg_pool_k=4,
                 innovate=False,
                 pool_after=False
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


        self.feature_dim    = feature_dim
        self.frame_size     = frame_size
        self.avg_pool_k     = avg_pool_k
        self.innovate       = innovate
        self.pool_after     = pool_after

        assert frame_size % avg_pool_k == 0
        self.env_dim = frame_size // avg_pool_k + 1

        # feature transform
        self.feature_alpha1 = nn.Conv1d(self.feature_dim + self.env_dim, frame_size, 2)
        self.feature_alpha2 = nn.Conv1d(frame_size, frame_size, 2)

        if self.innovate:
            self.feature_alpha1b = nn.Conv1d(self.feature_dim + self.env_dim, frame_size, 2)
            self.feature_alpha1c = nn.Conv1d(self.feature_dim + self.env_dim, frame_size, 2)

            self.feature_alpha2b = nn.Conv1d(frame_size, frame_size, 2)
            self.feature_alpha2c = nn.Conv1d(frame_size, frame_size, 2)


    def flop_count(self, rate):

        frame_rate = rate / self.frame_size

        shape_flops = sum([_conv1d_flop_count(x, frame_rate) for x in (self.feature_alpha1, self.feature_alpha2)]) + 11 * frame_rate * self.frame_size

        if self.innovate:
            inno_flops = sum([_conv1d_flop_count(x, frame_rate) for x in (self.feature_alpha1b, self.feature_alpha2b, self.feature_alpha1c, self.feature_alpha2c)]) + 22 * frame_rate * self.frame_size
        else:
            inno_flops = 0

        return shape_flops + inno_flops

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

    def forward(self, x, features, state=None, return_state=False, debug=False):
        """ innovate signal parts with temporal shaping


        Parameters:
        -----------
        x : torch.tensor
            input signal of shape (batch_size, 1, num_samples)

        features : torch.tensor
            frame-wise features of shape (batch_size, num_frames, feature_dim)

        """


        batch_size = x.size(0)
        num_frames = features.size(1)
        num_samples = x.size(2)

        # generate temporal envelope
        tenv = self.envelope_transform(x)

        # feature path
        f = torch.cat((features, tenv), dim=-1).permute(0, 2, 1)
        if state is not None:
            f = torch.cat((state, f), dim=-1)
        else:
            f = F.pad(f, [2, 0])
        alpha = F.leaky_relu(self.feature_alpha1(f), 0.2)
        alpha = torch.exp(self.feature_alpha2(alpha))
        alpha = alpha.permute(0, 2, 1)

        if self.innovate:
            inno_alpha = F.leaky_relu(self.feature_alpha1b(f), 0.2)
            inno_alpha = torch.exp(self.feature_alpha2b(inno_alpha))
            inno_alpha = inno_alpha.permute(0, 2, 1)

            inno_x = F.leaky_relu(self.feature_alpha1c(f), 0.2)
            inno_x = torch.tanh(self.feature_alpha2c(inno_x))
            inno_x = inno_x.permute(0, 2, 1)

        # signal path
        y = x.reshape(batch_size, num_frames, -1)
        y = alpha * y

        if self.innovate:
            y = y + inno_alpha * inno_x

        y = y.reshape(batch_size, 1, num_samples)

        if return_state:
            new_state = f[..., -2:]
            return y, new_state
        else:
            return y
