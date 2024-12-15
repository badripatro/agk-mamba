# Databricks notebook source
import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.fft import rfft, irfft, fft,ifft
from einops import rearrange

# functions

def exists(val):
    return val is not None

# classes

class SpectralKernel(nn.Module):
    def __init__(self,  dim):
        super().__init__()
        # this weights are valid for h=14 and w=8
        
        if dim == 64: #96 for large model, 64 for small and base model
            self.h = 56 #H
            self.w = 56 #(W/2)+1            
            self.complex_weight = nn.Parameter(torch.randn(self.h * self.w, dim, 2, dtype=torch.float32) * 0.02)
        if dim ==128:
            self.h = 28 #H
            self.w = 28 #(W/2)+1, this is due to rfft2
            self.complex_weight = nn.Parameter(torch.randn(self.h * self.w, dim, 2, dtype=torch.float32) * 0.02)
        if dim == 320: #96 for large model, 64 for small and base model
            self.h = 14 #H
            self.w = 14 #(W/2)+1            
            self.complex_weight = nn.Parameter(torch.randn(self.h * self.w, dim, 2, dtype=torch.float32) * 0.02)
        if dim ==448:
            self.h = 7 #H
            self.w = 7 #(W/2)+1, this is due to rfft2
            self.complex_weight = nn.Parameter(torch.randn(self.h * self.w, dim, 2, dtype=torch.float32) * 0.02)
        # This is for huge different from large
        if dim == 96: #96 for large model, 64 for small and base model
            self.h = 56 #H
            self.w = 56 #(W/2)+1            
            self.complex_weight = nn.Parameter(torch.randn(self.h * self.w, dim, 2, dtype=torch.float32) * 0.02)
        if dim ==192:
            self.h = 28 #H
            self.w = 28 #(W/2)+1, this is due to rfft2
            self.complex_weight = nn.Parameter(torch.randn(self.h * self.w, dim, 2, dtype=torch.float32) * 0.02)
        if dim == 384: #96 for large model, 64 for small and base model
            self.h = 14 #H
            self.w = 14 #(W/2)+1            
            self.complex_weight = nn.Parameter(torch.randn(self.h * self.w, dim, 2, dtype=torch.float32) * 0.02)
        if dim ==512:
            self.h = 7 #H
            self.w = 7 #(W/2)+1, this is due to rfft2
            self.complex_weight = nn.Parameter(torch.randn(self.h * self.w, dim, 2, dtype=torch.float32) * 0.02)

    def forward(self, x):
        # print('wno',x.shape) #CIFAR100 image :[128, 196, 384]
        B, N, C = x.shape 
        # print('wno B, N, C',B, N, C) #CIFAR100 image : 128 196 384
        # x = x.view(B, H, W, C)
        # B, H, W, C=x.shape
        x = x.to(torch.float32) 
        # print(x.dtype)
        # Add above for this error, RuntimeError: Input type (torch.cuda.HalfTensor) and weight type (torch.cuda.FloatTensor) should be the same
        x = torch.fft.fft2(x, dim=(1, 2), norm='ortho')
        # print('fgn',x.shape)
        weight = torch.view_as_complex(self.complex_weight)
        weight = torch.fft.fft2(weight, dim=(0,1), norm='ortho')

        # print('weight',weight.shape)
        x = x * weight
        x = torch.fft.ifft2(x, s=(N,C), dim=(1, 2), norm='ortho')
        # print('wno',x.shape)
        x = x.reshape(B, N, C)# permute is not same as reshape or view
        x = x.to(torch.float32) 
        return x

class GKSS(nn.Module):
    """ Pseudocode 3.2 """

    def __init__(
        self,
        *,
        dim,
        dim_expansion_factor = 4,
        kss_kernel_N = 512,
        kss_kernel_H = 256,
        reverse_seq = False,
        kss_kernel_lambda_imag_exp = True
    ):
        super().__init__()
        self.reverse_seq = reverse_seq
        self.norm = nn.LayerNorm(dim)

        dim_hidden = int(dim_expansion_factor * dim)
        self.to_u = nn.Sequential(nn.Linear(dim, dim_hidden, bias = False), nn.GELU())
        self.to_v = nn.Sequential(nn.Linear(dim, dim, bias = False), nn.GELU())

        self.kss = SpectralKernel(dim = dim)

        self.to_gate = nn.Linear(dim, dim_hidden, bias = False)
        self.to_out = nn.Linear(dim_hidden, dim)
        # self.act=nn.SiLU()

    def forward(self, x):
        if self.reverse_seq:
            x = torch.flip(x, dims = (1,))

        residual, x = x.clone(), self.norm(x)

        u = self.to_u(x)
        v = self.to_v(x)

        v = self.kss(v)

        uc = self.to_gate(v)
        out = self.to_out(uc * u)

        out = out + residual

        if self.reverse_seq:
            out = torch.flip(out, dims = (1,))

        return out       

