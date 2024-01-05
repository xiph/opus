import torch

from dnntools.sparsification import GRUSparsifier, LinearSparsifier, Conv1dSparsifier, ConvTranspose1dSparsifier

def mark_for_sparsification(module, params):
    setattr(module, 'sparsify', True)
    setattr(module, 'sparsification_params', params)
    return module

def create_sparsifier(module, start, stop, interval):
    sparsifier_list = []
    for m in module.modules():
        if hasattr(m, 'sparsify'):
            if isinstance(m, torch.nn.GRU):
                sparsifier_list.append(
                    GRUSparsifier([(m, m.sparsification_params)], start, stop, interval)
                )
            elif isinstance(m, torch.nn.Linear):
                sparsifier_list.append(
                    LinearSparsifier([(m, m.sparsification_params)], start, stop, interval)
                )
            elif isinstance(m, torch.nn.Conv1d):
                sparsifier_list.append(
                    Conv1dSparsifier([(m, m.sparsification_params)], start, stop, interval)
                )
            elif isinstance(m, torch.nn.ConvTranspose1d):
                sparsifier_list.append(
                    ConvTranspose1dSparsifier([(m, m.sparsification_params)], start, stop, interval)
                )
            else:
                print(f"[create_sparsifier] warning: module {m} marked for sparsification but no suitable sparsifier exists.")

    def sparsify(verbose=False):
        for sparsifier in sparsifier_list:
            sparsifier.step(verbose)

    return sparsify

def estimate_parameters(module):
    num_zero_parameters = 0
    if hasattr(module, 'sparsify'):
        if isinstance(module, torch.nn.Conv1d):
            pass
