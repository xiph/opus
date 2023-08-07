"""
Utility functions that are commonly used
"""

import numpy as np
from scipy.signal import windows, lfilter
from prettytable import PrettyTable


# Source: https://gist.github.com/thongonary/026210fc186eb5056f2b6f1ca362d912
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def stft(x, w = 'boxcar', N = 320, H = 160):
    x = np.concatenate([x,np.zeros(N)])
    # win_custom = np.concatenate([windows.hann(80)[:40],np.ones(240),windows.hann(80)[40:]])
    return np.stack([np.fft.rfft(x[i:i + N]*windows.get_window(w,N)) for i in np.arange(0,x.shape[0]-N,H)])

def random_filter(x):
    # Randomly filter x with second order IIR filter with coefficients in between -3/8,3/8
    filter_coeff = np.random.uniform(low =  -3.0/8, high = 3.0/8, size = 4)
    b = [1,filter_coeff[0],filter_coeff[1]]
    a = [1,filter_coeff[2],filter_coeff[3]]
    return lfilter(b,a,x)
