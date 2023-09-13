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

class FWConv(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3):
        super(FWConv, self).__init__()

        torch.manual_seed(5)

        self.in_size = in_size
        self.kernel_size = kernel_size
        self.conv = weight_norm(nn.Linear(in_size*self.kernel_size, out_size, bias=False))
        self.glu = GLU(out_size)

        self.init_weights()

    def init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d)\
            or isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.orthogonal_(m.weight.data)

    def forward(self, x, state):
        xcat = torch.cat((state, x), -1)
        #print(x.shape, state.shape, xcat.shape, self.in_size, self.kernel_size)
        out = self.glu(torch.tanh(self.conv(xcat)))
        return out, xcat[:,self.in_size:]

class FARGANCond(nn.Module):
    def __init__(self, feature_dim=20, cond_size=256, pembed_dims=64):
        super(FARGANCond, self).__init__()

        self.feature_dim = feature_dim
        self.cond_size = cond_size

        self.pembed = nn.Embedding(256, pembed_dims)
        self.fdense1 = nn.Linear(self.feature_dim + pembed_dims, self.cond_size, bias=False)
        self.fconv1 = nn.Conv1d(self.cond_size, self.cond_size, kernel_size=3, padding='valid', bias=False)
        self.fconv2 = nn.Conv1d(self.cond_size, self.cond_size, kernel_size=3, padding='valid', bias=False)
        self.fdense2 = nn.Linear(self.cond_size, 80*4, bias=False)

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
    def __init__(self, subframe_size=40, nb_subframes=4, cond_size=256):
        super(FARGANSub, self).__init__()

        self.subframe_size = subframe_size
        self.nb_subframes = nb_subframes
        self.cond_size = cond_size
        
        #self.sig_dense1 = nn.Linear(4*self.subframe_size+self.passthrough_size+self.cond_size, self.cond_size, bias=False)
        self.fwc0 = FWConv(4*self.subframe_size+80, self.cond_size)
        self.sig_dense2 = nn.Linear(self.cond_size, self.cond_size, bias=False)
        self.gru1 = nn.GRUCell(self.cond_size, self.cond_size, bias=False)
        self.gru2 = nn.GRUCell(self.cond_size, self.cond_size, bias=False)
        self.gru3 = nn.GRUCell(self.cond_size, self.cond_size, bias=False)
        
        self.dense1_glu = GLU(self.cond_size)
        self.dense2_glu = GLU(self.cond_size)
        self.gru1_glu = GLU(self.cond_size)
        self.gru2_glu = GLU(self.cond_size)
        self.gru3_glu = GLU(self.cond_size)
        self.ptaps_dense = nn.Linear(4*self.cond_size, 5)
        
        self.sig_dense_out = nn.Linear(4*self.cond_size, self.subframe_size, bias=False)
        self.gain_dense_out = nn.Linear(4*self.cond_size, 1)


        self.apply(init_weights)

    def forward(self, cond, prev, exc_mem, phase, period, states, gain=None):
        device = exc_mem.device
        #print(cond.shape, prev.shape)
        
        dump_signal(prev, 'prev_in.f32')

        idx = 256-torch.clamp(period[:,None], min=self.subframe_size+2, max=254)
        rng = torch.arange(self.subframe_size+4, device=device)
        idx = idx + rng[None,:] - 2
        pred = torch.gather(exc_mem, 1, idx)
        pred = pred/(1e-5+gain)

        prev = prev/(1e-5+gain)
        dump_signal(prev, 'pitch_exc.f32')
        dump_signal(exc_mem, 'exc_mem.f32')

        tmp = torch.cat((cond, pred[:,2:-2], prev, phase), 1)

        #tmp = self.dense1_glu(torch.tanh(self.sig_dense1(tmp)))
        fwc0_out, fwc0_state = self.fwc0(tmp, states[3])
        dense2_out = self.dense2_glu(torch.tanh(self.sig_dense2(fwc0_out)))
        gru1_state = self.gru1(dense2_out, states[0])
        gru1_out = self.gru1_glu(gru1_state)
        gru2_state = self.gru2(gru1_out, states[1])
        gru2_out = self.gru2_glu(gru2_state)
        gru3_state = self.gru3(gru2_out, states[2])
        gru3_out = self.gru3_glu(gru3_state)
        gru3_out = torch.cat([gru1_out, gru2_out, gru3_out, dense2_out], 1)
        sig_out = torch.tanh(self.sig_dense_out(gru3_out))
        dump_signal(sig_out, 'exc_out.f32')
        taps = self.ptaps_dense(gru3_out)
        taps = .2*taps + torch.exp(taps)
        taps = taps / (1e-2 + torch.sum(torch.abs(taps), dim=-1, keepdim=True))
        dump_signal(taps, 'taps.f32')
        #fpitch = taps[:,0:1]*pred[:,:-4] + taps[:,1:2]*pred[:,1:-3] + taps[:,2:3]*pred[:,2:-2] + taps[:,3:4]*pred[:,3:-1] + taps[:,4:]*pred[:,4:]
        fpitch = pred[:,2:-2]

        pitch_gain = torch.exp(self.gain_dense_out(gru3_out))
        dump_signal(pitch_gain, 'pgain.f32')
        sig_out = (sig_out + pitch_gain*fpitch) * gain
        exc_mem = torch.cat([exc_mem[:,self.subframe_size:], sig_out], 1)
        dump_signal(sig_out, 'sig_out.f32')
        return sig_out, exc_mem, (gru1_state, gru2_state, gru3_state, fwc0_state)

class FARGAN(nn.Module):
    def __init__(self, subframe_size=40, nb_subframes=4, feature_dim=20, cond_size=256, passthrough_size=0, has_gain=False, gamma=None):
        super(FARGAN, self).__init__()

        self.subframe_size = subframe_size
        self.nb_subframes = nb_subframes
        self.frame_size = self.subframe_size*self.nb_subframes
        self.feature_dim = feature_dim
        self.cond_size = cond_size

        self.cond_net = FARGANCond(feature_dim=feature_dim, cond_size=cond_size)
        self.sig_net = FARGANSub(subframe_size=subframe_size, nb_subframes=nb_subframes, cond_size=cond_size)

    def forward(self, features, period, nb_frames, pre=None, states=None):
        device = features.device
        batch_size = features.size(0)

        phase_real, phase_imag = gen_phase_embedding(period[:, 3:-1], self.frame_size)
        #np.round(32000*phase.detach().numpy()).astype('int16').tofile('phase.sw')

        prev = torch.zeros(batch_size, self.subframe_size, device=device)
        exc_mem = torch.zeros(batch_size, 256, device=device)
        nb_pre_frames = pre.size(1)//self.frame_size if pre is not None else 0

        states = (
            torch.zeros(batch_size, self.cond_size, device=device),
            torch.zeros(batch_size, self.cond_size, device=device),
            torch.zeros(batch_size, self.cond_size, device=device),
            torch.zeros(batch_size, (4*self.subframe_size+80)*2, device=device)
        )

        sig = torch.zeros((batch_size, 0), device=device)
        cond = self.cond_net(features, period)
        if pre is not None:
            prev[:,:] = pre[:, self.frame_size-self.subframe_size : self.frame_size]
            exc_mem[:,-self.frame_size:] = pre[:, :self.frame_size]
        start = 1 if nb_pre_frames>0 else 0
        for n in range(start, nb_frames+nb_pre_frames):
            for k in range(self.nb_subframes):
                pos = n*self.frame_size + k*self.subframe_size
                preal = phase_real[:, pos:pos+self.subframe_size]
                pimag = phase_imag[:, pos:pos+self.subframe_size]
                phase = torch.cat([preal, pimag], 1)
                #print("now: ", preal.shape, prev.shape, sig_in.shape)
                pitch = period[:, 3+n]
                gain = .03*10**(0.5*features[:, 3+n, 0:1]/np.sqrt(18.0))
                #gain = gain[:,:,None]
                out, exc_mem, states = self.sig_net(cond[:, n, k*80:(k+1)*80], prev, exc_mem, phase, pitch, states, gain=gain)

                if n < nb_pre_frames:
                    out = pre[:, pos:pos+self.subframe_size]
                    exc_mem[:,-self.subframe_size:] = out
                else:
                    sig = torch.cat([sig, out], 1)

                prev = out
        states = [s.detach() for s in states]
        return sig, states

