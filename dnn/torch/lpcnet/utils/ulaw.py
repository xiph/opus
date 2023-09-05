import math as m

import torch



def ulaw2lin(u):
    scale_1 = 32768.0 / 255.0
    u = u - 128
    s = torch.sign(u)
    u = torch.abs(u)
    return s * scale_1 * (torch.exp(u / 128. * m.log(256)) - 1)


def lin2ulawq(x):
    scale = 255.0 / 32768.0
    s = torch.sign(x)
    x = torch.abs(x)
    u = s * (128 * torch.log(1 + scale * x) / m.log(256))
    u = torch.clip(128 + torch.round(u), 0, 255)
    return u

def lin2ulaw(x):
    scale = 255.0 / 32768.0
    s = torch.sign(x)
    x = torch.abs(x)
    u = s * (128 * torch.log(1 + scale * x) / torch.log(256))
    u = torch.clip(128 + u, 0, 255)
    return u