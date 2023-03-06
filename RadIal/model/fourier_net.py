import torch
import numpy as np
from cplxmodule.nn import CplxConv1d, CplxLinear, CplxDropout
from cplxmodule.nn import CplxModReLU, CplxParameter, CplxModulus, CplxToCplx, CplxAngle
from cplxmodule.nn.modules.casting import TensorToCplx,CplxToTensor
from cplxmodule.nn import RealToCplx, CplxToReal
import torch.nn as nn
import torch.nn.functional as F

class bblock(nn.Module):
    def __init__(self,in_chans,out_chans,stride,kernel_size,padding):
        super(bblock,self).__init__()
        self.conv = nn.Conv2d(in_chans,out_chans,kernel_size=kernel_size,stride=stride,padding=padding)
        self.activation = nn.ReLU()
        self.bnorm = nn.BatchNorm2d(out_chans)

    def forward(self,x):
        x = self.conv(x)
        x = self.bnorm(x)
        x = self.activation(x)

        return x


class inverse_bblock(nn.Module):
    def __init__(self,in_chans,out_chans,stride,kernel_size,padding):
        super(inverse_bblock,self).__init__()
        self.conv = nn.ConvTranspose2d(in_chans,out_chans,kernel_size=kernel_size,stride=stride,padding=padding)
        self.activation = nn.ReLU()
        self.bnorm = nn.BatchNorm2d(out_chans)

    def forward(self,x):
        x = self.conv(x)
        x = self.bnorm(x)
        x = self.activation(x)

        return x

class outblock_bblock(nn.Module):
    def __init__(self,in_chans,out_chans,stride,kernel_size,padding):
        super(outblock_bblock,self).__init__()
        self.conv = nn.ConvTranspose2d(in_chans,out_chans,kernel_size=kernel_size,stride=stride,padding=padding)
        self.activation = nn.ELU()
        self.bnorm = nn.BatchNorm2d(out_chans)

    def forward(self,x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class DFT_FCN(nn.Module):
    def __init__(self,):
        super(DFT_FCN,self).__init__()

        # Encoder
        self.enc1 = bblock(16,32,kernel_size=(4,2),stride=4,padding=(1,0))
        self.enc2 = bblock(32,64,kernel_size=(4,2),stride=4,padding=(1,0))
        self.enc3 = bblock(64,128,kernel_size=(4,2),stride=2,padding=(1,0))
        self.enc4 = bblock(128,256,kernel_size=(4,2),stride=2,padding=(1,0))
        self.mu = nn.Linear(1024,1024)
        # Decoder
        self.deconv1 = inverse_bblock(512,256,kernel_size=(4,1),stride=2,padding=(1,0))
        self.deconv2 = inverse_bblock(256,128,kernel_size=(4,2),stride=2,padding=(1,0))
        self.deconv3 = inverse_bblock(128,64,kernel_size=(4,2),stride=2,padding=(1,0))
        self.deconv4 = inverse_bblock(64,32,kernel_size=(4,4),stride=4,padding=(0,0))
        self.deconv5 = outblock_bblock(32,16,kernel_size=(4,4),stride=4,padding=(0,0))
        #self.up1 = nn.Upsample(scale_factor=(2,1))

    def forward(self,x):
        x1 = self.enc1(x)
        #print(x1.shape)
        x2 = self.enc2(x1)
        #print(x2.shape)
        x3 = self.enc3(x2)
        #print(x3.shape)
        x4 = self.enc4(x3)
        #print(x4.shape)
        z = self.mu(torch.flatten(x4,1,3)).view(-1,512,2,1)
        #print(z.shape)
        #x5 = self.up1(z)

        x5 = self.deconv1(z)
        #print('Up1',x5.shape)
        x6 = self.deconv2(x5)
        #print(x6.shape)
        x7 = self.deconv3(x6)
        #print(x7.shape)
        x8 = self.deconv4(x7)
        #print(x8.shape)
        out = self.deconv5(x8)
        return out



class Range_Fourier_Net(nn.Module):
    def __init__(self):
        super(Range_Fourier_Net, self).__init__()
        self.range_nn = CplxLinear(512, 512, bias = False)
        range_weights = np.zeros((512, 512), dtype = np.complex64)
        for j in range(0, 512):
            for h in range(0, 512):
                range_weights[h][j] = np.exp(-1j * 2 * np.pi *(j*h/512))
        range_weights = TensorToCplx()(torch.view_as_real(torch.from_numpy(range_weights)))
        self.range_nn.weight = CplxParameter(range_weights)

    def forward(self, x):
        x = self.range_nn(x)
        return x

class Doppler_Fourier_Net(nn.Module):
    def __init__(self):
        super(Doppler_Fourier_Net, self).__init__()
        self.doppler_nn = CplxLinear(256, 256, bias = False)
        doppler_weights = np.zeros((256, 256), dtype=np.complex64)
        for j in range(0, 256):
            for h in range(0, 256):
                hh = h + 128
                if hh >= 256:
                    hh = hh - 256
                doppler_weights[h][j] = np.exp(-1j * 2 * np.pi * (j * hh /256))
        doppler_weights = TensorToCplx()(torch.view_as_real(torch.from_numpy(doppler_weights)))
        self.doppler_nn.weight = CplxParameter(doppler_weights)

    def forward(self, x):
        x = self.doppler_nn(x)
        return x


class NoShift_Doppler_Fourier_Net(nn.Module):
    def __init__(self):
        super(NoShift_Doppler_Fourier_Net, self).__init__()
        self.doppler_nn = CplxLinear(256, 256, bias = False)
        doppler_weights = np.zeros((256, 256), dtype=np.complex64)
        for j in range(0, 256):
            for h in range(0, 256):
                doppler_weights[h][j] = np.exp(-1j * 2 * np.pi * (j * h /256))
        doppler_weights = TensorToCplx()(torch.view_as_real(torch.from_numpy(doppler_weights)))
        self.doppler_nn.weight = CplxParameter(doppler_weights)

    def forward(self, x):
        x = self.doppler_nn(x)
        return x

class ComplexAct(nn.Module):
    def __init__(self, act, use_phase=False):
        super(ComplexAct,self).__init__()
        # act can be either a function from nn.functional or a nn.Module if the
        # activation has learnable parameters
        self.act = act
        self.use_phase = use_phase
        self.mod = CplxModulus()
        self.angle = CplxAngle()
        self.m = nn.Parameter(torch.tensor(0.0).float())
        self.m.requires_grad = True

    def forward(self, z):
        if self.use_phase:
            return self.act(self.mod(z) + self.m) * torch.exp(1.j * self.angle(z))
        else:
            return self.act(z.real) + 1.j * self.act(z.imag)

class FFT_Net(nn.Module):
    def __init__(self,):
        super(FFT_Net,self).__init__()
        self.range_net = Range_Fourier_Net()
        self.doppler_net = NoShift_Doppler_Fourier_Net()
        self.cplx_transpose = CplxToCplx[torch.transpose]
        self.norm = nn.InstanceNorm2d(32)
        #self.activation = ComplexAct(act=nn.LeakyReLU(),use_phase=True)

    def forward(self,x):
        x = x.permute(0,1,3,2)
        x = self.range_net(x)
        x = self.cplx_transpose(2,3)(x)
        x = self.doppler_net(x)
        x = torch.concat([x.real,x.imag],axis=1)
        out = self.norm(x)
        return out

#    def forward(self,x):
#        x = x.permute(0,1,3,2)
#        x = self.range_net(x)
#        x = self.activation(x).unsqueeze(-1)
#        x = TensorToCplx()(torch.concat([x.real,x.imag],axis=-1))
#        x = self.cplx_transpose(2,3)(x)
#        x = self.doppler_net(x)
#        x = self.activation(x)
#        x = torch.concat([x.real,x.imag],axis=1)
#        out = self.norm(x)
#        return out
