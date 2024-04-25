import torch
import scipy.signal


from utils.layers.fir import FIR

class TDLowpass(torch.nn.Module):
    def __init__(self, numtaps, cutoff, power=2):
        super().__init__()

        self.b = scipy.signal.firwin(numtaps, cutoff)
        self.weight = torch.nn.Parameter(torch.from_numpy(self.b).float().view(1, 1, -1), requires_grad=False)
        self.power = power

    def forward(self, y_true, y_pred):

        if len(y_true.shape) < 3: y_true = y_true.unsqueeze(1)
        if len(y_pred.shape) < 3: y_pred = y_pred.unsqueeze(1)

        diff = y_true - y_pred
        diff_lp = torch.nn.functional.conv1d(diff, self.weight)

        loss = torch.mean(torch.abs(diff_lp) ** self.power) / (torch.mean(torch.abs(y_true) ** self.power) + 1e-6**self.power)
        loss = loss ** 1/self.power

        return loss

    def get_freqz(self):
        freq, response = scipy.signal.freqz(self.b)

        return freq, response
