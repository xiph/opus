""" module implementing PCM embeddings for LPCNet """

import math as m

import torch
from torch import nn


class PCMEmbedding(nn.Module):
    def __init__(self, embed_dim=128, num_levels=256):
        super(PCMEmbedding, self).__init__()

        self.embed_dim  = embed_dim
        self.num_levels = num_levels

        self.embedding = nn.Embedding(self.num_levels, self.num_dim)

        # initialize
        with torch.no_grad():
            num_rows, num_cols = self.num_levels, self.embed_dim
            a = m.sqrt(12) * (torch.rand(num_rows, num_cols) - 0.5)
            for i in range(num_rows):
                a[i, :] += m.sqrt(12) * (i -  num_rows / 2)
            self.embedding.weight[:, :] = 0.1 * a

    def forward(self, x):
        return self.embeddint(x)


class DifferentiablePCMEmbedding(PCMEmbedding):
    def __init__(self, embed_dim, num_levels=256):
        super(DifferentiablePCMEmbedding, self).__init__(embed_dim, num_levels)

    def forward(self, x):
        x_int = (x - torch.floor(x)).detach().long()
        x_frac = x - x_int
        x_next = torch.minimum(x_int + 1, self.num_levels)

        embed_0 = self.embedding(x_int)
        embed_1 = self.embedding(x_next)

        return (1 - x_frac) * embed_0 + x_frac * embed_1
