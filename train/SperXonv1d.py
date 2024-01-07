import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1,dtype=torch.cfloat))
    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bixz,ioyz->boxz", input, weights)

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)


    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)  #[1,1024,513]
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-1)//2 + 1, device=x.device, dtype=torch.float)#normalized=False[1,1024,513,2]
        out_ft[:, :, :self.modes1]=self.compl_mul2d(x_ft[:, :, :self.modes1], self.weights1)      #Return to physical space
        # x = torch.irfft(out_ft,2, n=x.size(-1))
        # x = torch.irfft(out_ft,2, normalized=True,onesided=True)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        x =x[:,:,:]
        return x

