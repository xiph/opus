import torch


def sample_excitation(probs, pitch_corr):

    norm = lambda x : x / (x.sum() + 1e-18)

    # lowering the temperature
    probs = norm(probs ** (1 + max(0, 1.5 * pitch_corr - 0.5)))
    # cut-off tails
    probs = norm(torch.maximum(probs - 0.002 , torch.FloatTensor([0])))
    # sample
    exc   = torch.multinomial(probs.squeeze(), 1)

    return exc
