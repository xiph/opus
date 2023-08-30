import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import filters
from torch.nn.utils import weight_norm

Fs = 16000

fid_dict = {}
def dump_signal(x, filename):
    return
    if filename in fid_dict:
        fid = fid_dict[filename]
    else:
        fid = open(filename, "w")
        fid_dict[filename] = fid
    x = x.detach().numpy().astype('float32')
    x.tofile(fid)


def sig_l1(y_true, y_pred):
    return torch.mean(abs(y_true-y_pred))/torch.mean(abs(y_true))

def sig_loss(y_true, y_pred):
    t = y_true/(1e-15+torch.norm(y_true, dim=-1, p=2, keepdim=True))
    p = y_pred/(1e-15+torch.norm(y_pred, dim=-1, p=2, keepdim=True))
    return torch.mean(1.-torch.sum(p*t, dim=-1))


def analysis_filter(x, lpc, nb_subframes=4, subframe_size=40, gamma=.9):
    device = x.device
    batch_size = lpc.size(0)

    nb_frames = lpc.shape[1]


    sig = torch.zeros(batch_size, subframe_size+16, device=device)
    x = torch.reshape(x, (batch_size, nb_frames*nb_subframes, subframe_size))
    out = torch.zeros((batch_size, 0), device=device)

    if gamma is not None:
        bw = gamma**(torch.arange(1, 17, device=device))
        lpc = lpc*bw[None,None,:]
    ones = torch.ones((*(lpc.shape[:-1]), 1), device=device)
    zeros = torch.zeros((*(lpc.shape[:-1]), subframe_size-1), device=device)
    a = torch.cat([ones, lpc], -1)
    a_big = torch.cat([a, zeros], -1)
    fir_mat_big = filters.toeplitz_from_filter(a_big)

    #print(a_big[:,0,:])
    for n in range(nb_frames):
        for k in range(nb_subframes):

            sig = torch.cat([sig[:,subframe_size:], x[:,n*nb_subframes + k, :]], 1)
            exc = torch.bmm(fir_mat_big[:,n,:,:], sig[:,:,None])
            out = torch.cat([out, exc[:,-subframe_size:,0]], 1)

    return out


# weight initialization and clipping
def init_weights(module):
    if isinstance(module, nn.GRU):
        for p in module.named_parameters():
            if p[0].startswith('weight_hh_'):
                nn.init.orthogonal_(p[1])

def gen_phase_embedding(periods, frame_size):
    device = periods.device
    batch_size = periods.size(0)
    nb_frames = periods.size(1)
    w0 = 2*torch.pi/periods
    w0_shift = torch.cat([2*torch.pi*torch.rand((batch_size, 1), device=device)/frame_size, w0[:,:-1]], 1)
    cum_phase = frame_size*torch.cumsum(w0_shift, 1)
    fine_phase = w0[:,:,None]*torch.broadcast_to(torch.arange(frame_size, device=device), (batch_size, nb_frames, frame_size))
    embed = torch.unsqueeze(cum_phase, 2) + fine_phase
    embed = torch.reshape(embed, (batch_size, -1))
    return torch.cos(embed), torch.sin(embed)

class GLU(nn.Module):
    def __init__(self, feat_size):
        super(GLU, self).__init__()
        
        torch.manual_seed(5)

        self.gate = weight_norm(nn.Linear(feat_size, feat_size, bias=False))

        self.init_weights()

    def init_weights(self):
    
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d)\
            or isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.orthogonal_(m.weight.data)

    def forward(self, x):
        
        out = x * torch.sigmoid(self.gate(x)) 
        
        return out


class FARGANCond(nn.Module):
    def __init__(self, feature_dim=20, cond_size=256, pembed_dims=64):
        super(FARGANCond, self).__init__()

        self.feature_dim = feature_dim
        self.cond_size = cond_size

        self.pembed = nn.Embedding(256, pembed_dims)
        self.fdense1 = nn.Linear(self.feature_dim + pembed_dims, self.cond_size, bias=False)
        self.fconv1 = nn.Conv1d(self.cond_size, self.cond_size, kernel_size=3, padding='valid', bias=False)
        self.fconv2 = nn.Conv1d(self.cond_size, self.cond_size, kernel_size=3, padding='valid', bias=False)
        self.fdense2 = nn.Linear(self.cond_size, self.cond_size, bias=False)

        self.apply(init_weights)

    def forward(self, features, period):
        p = self.pembed(period)
        features = torch.cat((features, p), -1)
        tmp = torch.tanh(self.fdense1(features))
        tmp = tmp.permute(0, 2, 1)
        tmp = torch.tanh(self.fconv1(tmp))
        tmp = torch.tanh(self.fconv2(tmp))
        tmp = tmp.permute(0, 2, 1)
        tmp = torch.tanh(self.fdense2(tmp))
        return tmp

class FARGANSub(nn.Module):
    def __init__(self, subframe_size=40, nb_subframes=4, cond_size=256, passthrough_size=0, has_gain=False):
        super(FARGANSub, self).__init__()

        self.subframe_size = subframe_size
        self.nb_subframes = nb_subframes
        self.cond_size = cond_size
        self.has_gain = has_gain
        self.passthrough_size = passthrough_size
        
        print("has_gain:", self.has_gain)
        print("passthrough_size:", self.passthrough_size)
        
        gain_param = 1 if self.has_gain else 0

        self.sig_dense1 = nn.Linear(3*self.subframe_size+self.passthrough_size+self.cond_size+gain_param, self.cond_size, bias=False)
        self.sig_dense2 = nn.Linear(self.cond_size, self.cond_size, bias=False)
        self.gru1 = nn.GRUCell(self.cond_size, self.cond_size, bias=False)
        self.gru2 = nn.GRUCell(self.cond_size, self.cond_size, bias=False)
        self.gru3 = nn.GRUCell(self.cond_size, self.cond_size, bias=False)
        
        self.dense1_glu = GLU(self.cond_size)
        self.dense2_glu = GLU(self.cond_size)
        self.gru1_glu = GLU(self.cond_size)
        self.gru2_glu = GLU(self.cond_size)
        self.gru3_glu = GLU(self.cond_size)
        
        self.sig_dense_out = nn.Linear(self.cond_size, self.subframe_size+self.passthrough_size, bias=False)
        if self.has_gain:
            self.gain_dense_out = nn.Linear(self.cond_size, 1)


        self.apply(init_weights)

    def forward(self, cond, prev, exc_mem, phase, period, states):
        device = exc_mem.device
        #print(cond.shape, prev.shape)
        
        dump_signal(prev, 'prev_in.f32')

        idx = 256-torch.maximum(torch.tensor(self.subframe_size, device=device), period[:,None])
        rng = torch.arange(self.subframe_size, device=device)
        idx = idx + rng[None,:]
        prev = torch.gather(exc_mem, 1, idx)
        #prev = prev*0
        dump_signal(prev, 'pitch_exc.f32')
        dump_signal(exc_mem, 'exc_mem.f32')
        if self.has_gain:
            gain = torch.norm(prev, dim=1, p=2, keepdim=True)
            prev = prev/(1e-5+gain)
            prev = torch.cat([prev, torch.log(1e-5+gain)], 1)

        passthrough = states[3]
        tmp = torch.cat((cond, prev, passthrough, phase), 1)

        tmp = self.dense1_glu(torch.tanh(self.sig_dense1(tmp)))
        tmp = self.dense2_glu(torch.tanh(self.sig_dense2(tmp)))
        gru1_state = self.gru1(tmp, states[0])
        gru2_state = self.gru2(self.gru1_glu(gru1_state), states[1])
        gru3_state = self.gru3(self.gru2_glu(gru2_state), states[2])
        gru3_out = self.gru3_glu(gru3_state)
        sig_out = torch.tanh(self.sig_dense_out(gru3_out))
        if self.passthrough_size != 0:
            passthrough = sig_out[:,self.subframe_size:]
            sig_out = sig_out[:,:self.subframe_size]
        if self.has_gain:
            out_gain = torch.exp(self.gain_dense_out(gru3_out))
            sig_out = sig_out * out_gain
        dump_signal(sig_out, 'exc_out.f32')
        exc_mem = torch.cat([exc_mem[:,self.subframe_size:], sig_out], 1)
        dump_signal(sig_out, 'sig_out.f32')
        return sig_out, exc_mem, (gru1_state, gru2_state, gru3_state, passthrough)

class FARGAN(nn.Module):
    def __init__(self, subframe_size=40, nb_subframes=4, feature_dim=20, cond_size=256, passthrough_size=0, has_gain=False, gamma=None):
        super(FARGAN, self).__init__()

        self.subframe_size = subframe_size
        self.nb_subframes = nb_subframes
        self.frame_size = self.subframe_size*self.nb_subframes
        self.feature_dim = feature_dim
        self.cond_size = cond_size
        self.has_gain = has_gain
        self.passthrough_size = passthrough_size

        self.cond_net = FARGANCond(feature_dim=feature_dim, cond_size=cond_size)
        self.sig_net = FARGANSub(subframe_size=subframe_size, nb_subframes=nb_subframes, cond_size=cond_size, has_gain=has_gain, passthrough_size=passthrough_size)

    def forward(self, features, period, nb_frames, pre=None, states=None):
        device = features.device
        batch_size = features.size(0)

        phase_real, phase_imag = gen_phase_embedding(period[:, 3:-1], self.frame_size)
        #np.round(32000*phase.detach().numpy()).astype('int16').tofile('phase.sw')

        prev = torch.zeros(batch_size, self.subframe_size, device=device)
        exc_mem = torch.zeros(batch_size, 256, device=device)
        nb_pre_frames = pre.size(1)//self.frame_size if pre is not None else 0

        if states is None:
            states = (
                torch.zeros(batch_size, self.cond_size, device=device),
                torch.zeros(batch_size, self.cond_size, device=device),
                torch.zeros(batch_size, self.cond_size, device=device),
                torch.zeros(batch_size, self.passthrough_size, device=device)
            )

        sig = torch.zeros((batch_size, 0), device=device)
        cond = self.cond_net(features, period)
        passthrough = torch.zeros(batch_size, self.passthrough_size, device=device)
        for n in range(nb_frames+nb_pre_frames):
            for k in range(self.nb_subframes):
                pos = n*self.frame_size + k*self.subframe_size
                preal = phase_real[:, pos:pos+self.subframe_size]
                pimag = phase_imag[:, pos:pos+self.subframe_size]
                phase = torch.cat([preal, pimag], 1)
                #print("now: ", preal.shape, prev.shape, sig_in.shape)
                pitch = period[:, 3+n]
                out, exc_mem, states = self.sig_net(cond[:, n, :], prev, exc_mem, phase, pitch, states)

                if n < nb_pre_frames:
                    out = pre[:, pos:pos+self.subframe_size]
                    exc_mem[:,-self.subframe_size:] = out
                else:
                    sig = torch.cat([sig, out], 1)

                prev = out
        states = [s.detach() for s in states]
        return sig, states

