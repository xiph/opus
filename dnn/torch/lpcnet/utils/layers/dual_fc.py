import torch
from torch import nn

class DualFC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DualFC, self).__init__()

        self.dense1 = nn.Linear(input_dim, output_dim)
        self.dense2 = nn.Linear(input_dim, output_dim)

        self.alpha = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.beta  = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

    def forward(self, x):
        return self.alpha * torch.tanh(self.dense1(x)) + self.beta * torch.tanh(self.dense2(x))
