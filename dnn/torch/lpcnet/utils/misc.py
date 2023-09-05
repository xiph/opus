import torch


def find(a, v):
    try:
        idx = a.index(v)
    except:
        idx = -1
    return idx

def interleave_tensors(tensors, dim=-2):
    """ interleave list of tensors along sequence dimension """

    x = torch.cat([x.unsqueeze(dim) for x in tensors], dim=dim)
    x = torch.flatten(x, dim - 1, dim)

    return x

def _interleave(x, pcm_levels=256):

    repeats = pcm_levels // (2*x.size(-1))
    x = x.unsqueeze(-1)
    p = torch.flatten(torch.repeat_interleave(torch.cat((x, 1 - x), dim=-1), repeats, dim=-1), -2)

    return p

def get_pdf_from_tree(x):
    pcm_levels = x.size(-1)

    p = _interleave(x[..., 1:2])
    n = 4
    while n <= pcm_levels:
        p = p * _interleave(x[..., n//2:n])
        n *= 2

    return p