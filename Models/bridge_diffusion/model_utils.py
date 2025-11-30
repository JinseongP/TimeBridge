import math
import scipy
import torch
import torch.nn.functional as F
import numpy as np
import gpytorch

from torch import nn, einsum
from functools import partial
from einops import rearrange, reduce
from scipy.fftpack import next_fast_len
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA



def filter_parameters_calculation(filter_strength, seq_length=-1):
    if seq_length != -1:    
        filter_period = seq_length
    else:
        filter_period = 7  # default filter period

    # use the min_length_limit to calculate the Wn_key_indicator and Wn_pct
    # the max Wn_key_indicator is 2, and the max Wn_pct is 2 for not filtering
    Wn_key_indicator = min(2 / (filter_period * filter_strength), 2)
    # Wn_pct=min(2,2/(filter_period*Wn_pct_factor))
    return Wn_key_indicator

def butter_lowpass_filter(x, order=4, device='cpu'):
    (batch_size, seq_length, feature_size) = x.shape
    x_ = x.transpose(1, 2).reshape(-1,seq_length).cpu().detach().numpy()
    
    Wn = filter_parameters_calculation(filter_strength=1)
    b, a = butter(order, Wn, btype='low', analog=False)
    y_ = torch.from_numpy(filtfilt(b, a, x_).copy()).float().to(device)
    y = y_.reshape(-1, feature_size, seq_length)
    y = y.transpose(1,2)
    y = torch.clip(y, min=-1, max=1)
    return y


def poly_approx(x, degree=3, device='cpu'):
    batch_size, seq_length, feature_size = x.shape
    x_ = x.transpose(1, 2).reshape(-1,seq_length).cpu().detach().numpy()
    time = [i for i in range(1, seq_length+1)]

    outputs_ = np.polyfit(np.array(time), np.transpose(x_), degree, cov=False)
    outputs = np.transpose(outputs_) # len(data)*seq_size, coef_size
    coeff = np.array(outputs).reshape(-1, feature_size, degree+1)
    coeff = torch.tensor(coeff).transpose(1,2).to(device).float()

    polys = []
    for i in range(len(outputs)):
        poly_ = np.poly1d(outputs[i])
        polys.append([poly_(i) for i in time])

    poly_prior_ = np.array(polys).reshape(-1, feature_size, seq_length)
    poly_prior = torch.tensor(poly_prior_).transpose(1,2).to(device).float()

    return poly_prior


def gp(x, kernel_type='rbf', bw='GAUSSIAN', var=1.0, device='cpu'):
    batch_size, seq_length, feature_size = x.shape
    mean = x.float().mean(dim=0).transpose(0,1) #feature_size*seq_length
    stds = x.float().std(dim=0) #seq_length*feature_size
    l = x.reshape(batch_size,-1).float().std()
    
    indices = torch.arange(seq_length)
    distances = torch.abs(indices.view(-1, 1) - indices.view(1, -1)).float()
    
    if kernel_type == 'rbf':
        kernel = torch.exp(-distances**2 / (2*(l**2)))
        
    elif kernel_type == 'epanechnikov':
        kernel = 0.75*torch.max(1 - distances**2, torch.zeros_like(distances))

    base_kernel = var * kernel
    kernels = []
    for i in range(feature_size):
        diag_addition = torch.diag(stds[:, i])
        kernel = base_kernel + diag_addition.unsqueeze(0)
        kernels.append(kernel)
    
    kernels = torch.stack(kernels, dim=0).squeeze(1).to(device).float()
    return mean, kernels

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding=1)
    )

def Downsample(dim, dim_out=None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)


def get_dev():
    """
    Get the device to use for torch.distributed.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def normalize_to_neg_one_to_one(x):
    return x * 2 - 1

def unnormalize_to_zero_to_one(x):
    return (x + 1) * 0.5

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = torch.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# learnable positional embeds

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(1, max_len, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        """
        # print(x.shape)
        x = x + self.pe
        return self.dropout(x)


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1-torch.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, torch.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean=[]
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean=torch.cat(moving_mean,dim=-1)
        moving_mean = torch.sum(moving_mean*nn.Softmax(-1)(self.layer(x.unsqueeze(-1))),dim=-1)
        res = x - moving_mean
        return res, moving_mean 


class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module. """
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.transpose(*self.shape)
    

class Conv_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, resid_pdrop=0.):
        super().__init__()
        self.sequential = nn.Sequential(
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_dim, out_dim, 3, stride=1, padding=1),
            nn.Dropout(p=resid_pdrop),
        )

    def forward(self, x):
        return self.sequential(x).transpose(1, 2)
    

class Transformer_MLP(nn.Module):
    def __init__(self, n_embd, mlp_hidden_times, act, resid_pdrop):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels=n_embd, out_channels=int(mlp_hidden_times * n_embd), kernel_size=1, padding=0),
            act,
            nn.Conv1d(in_channels=int(mlp_hidden_times * n_embd), out_channels=int(mlp_hidden_times * n_embd), kernel_size=3, padding=1),
            act,
            nn.Conv1d(in_channels=int(mlp_hidden_times * n_embd), out_channels=n_embd,  kernel_size=3, padding=1),
            nn.Dropout(p=resid_pdrop),
        )

    def forward(self, x):
        return self.sequential(x)
    

class GELU2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * F.sigmoid(1.702 * x)


class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.emb = SinusoidalPosEmb(n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd*2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x, timestep, label_emb=None):
        emb = self.emb(timestep)
        if label_emb is not None:
            emb = emb + label_emb
        emb = self.linear(self.silu(emb)).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x
    

class AdaInsNorm(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.emb = SinusoidalPosEmb(n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd*2)
        self.instancenorm = nn.InstanceNorm1d(n_embd)

    def forward(self, x, timestep, label_emb=None):
        emb = self.emb(timestep)
        if label_emb is not None:
            emb = emb + label_emb
        emb = self.linear(self.silu(emb)).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.instancenorm(x.transpose(-1, -2)).transpose(-1,-2) * (1 + scale) + shift
        return x


    