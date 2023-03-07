import torch
import numpy as np
from cplxmodule.nn import CplxConv1d, CplxLinear, CplxDropout
from cplxmodule.nn import CplxModReLU, CplxParameter, CplxModulus, CplxToCplx, CplxAngle
from cplxmodule.nn.modules.casting import TensorToCplx,CplxToTensor
from cplxmodule.nn import RealToCplx, CplxToReal
import torch.nn as nn
import torch.nn.functional as F

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
