import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import Sequential
from torchvision.transforms.transforms import Sequence
from functools import partial
from model.Swin import SwinTransformer
from model.fourier_net import FFT_Net
import numpy as np

NbTxAntenna = 12
NbRxAntenna = 16
NbVirtualAntenna = NbTxAntenna * NbRxAntenna

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

class Detection_Header(nn.Module):

    def __init__(self, use_bn=True,reg_layer=2,input_angle_size=128):
        super(Detection_Header, self).__init__()

        self.use_bn = use_bn
        self.reg_layer = reg_layer
        self.input_angle_size = input_angle_size
        self.target_angle = 128
        bias = not use_bn

        if(self.input_angle_size==128):
            self.conv1 = conv3x3(128, 64, bias=bias)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = conv3x3(64, 32, bias=bias)
            self.bn2 = nn.BatchNorm2d(32)
        elif(self.input_angle_size==448):
            self.conv1 = conv3x3(256, 144, bias=bias,stride=(1,2))
            self.bn1 = nn.BatchNorm2d(144)
            self.conv2 = conv3x3(144, 96, bias=bias)
            self.bn2 = nn.BatchNorm2d(96)
        elif(self.input_angle_size==896):
            self.conv1 = conv3x3(256, 144, bias=bias,stride=(1,2))
            self.bn1 = nn.BatchNorm2d(144)
            self.conv2 = conv3x3(144, 96, bias=bias,stride=(1,2))
            self.bn2 = nn.BatchNorm2d(96)
        else:
            raise NameError('Wrong channel angle paraemter !')
            return

        self.conv3 = conv3x3(32, 32, bias=bias)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = conv3x3(32, 32, bias=bias)
        self.bn4 = nn.BatchNorm2d(32)

        self.clshead = conv3x3(32, 1, bias=True)
        self.reghead = conv3x3(32, reg_layer, bias=True)
        self.objhead = conv3x3(32,6,bias=True)

    def forward(self, x):

        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.conv3(x)
        if self.use_bn:
            x = self.bn3(x)
        x = self.conv4(x)
        if self.use_bn:
            x = self.bn4(x)

        cls = torch.sigmoid(self.clshead(x))
        reg = self.reghead(x)
        obj = self.objhead(x)

        return torch.cat([cls, reg], dim=1),obj


class Bottleneck(nn.Module):

    def __init__(self, in_planes, planes, stride=1, downsample=None,expansion=4):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(expansion*planes)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = F.relu(residual + out)
        return out


class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            out = self.downsample(out)

        return out

class block(nn.Module):
    def __init__(self,in_chans,out_chans,stride,kernel_size):
        super(block,self).__init__()
        self.deconv = nn.ConvTranspose2d(in_chans,out_chans,kernel_size=kernel_size,stride=stride,padding=0)
        self.activation = nn.ReLU()
        self.bnorm = nn.BatchNorm2d(out_chans)

    def forward(self,x):
        x = self.deconv(x)
        x = self.bnorm(x)
        x = self.activation(x)

        return x

class ViT_RangeAngle_Decoder(nn.Module):
    def __init__(self):
        super(ViT_RangeAngle_Decoder,self).__init__()
        self.blk1 = block(2,32,2,2)
        self.blk2 = block(32,64,2,2)
        self.blk3 = block(64,256,2,2)
        self.dense = nn.Linear(512,896)
        self.dense_acc = nn.ReLU()

    def forward(self,x):
        x = self.dense(x)
        x = self.dense_acc(x).reshape(-1,2,16,28)

        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)

        return x

class Swin_RangeAngle_Decoder(nn.Module):
    def __init__(self, ):
        super(Swin_RangeAngle_Decoder, self).__init__()

        # Top-down layers
        self.deconv4 = nn.ConvTranspose2d(4, 4, kernel_size=3, stride=(2,1), padding=1, output_padding=(1,0))

        self.conv_block4 = BasicBlock(12,64)
        self.deconv3 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=(2,1), padding=1, output_padding=(1,0))
        self.conv_block3 = BasicBlock(80,128)
        self.deconv2 = nn.ConvTranspose2d(80,80,kernel_size=3,stride=(2,1),padding=1,output_padding=(1,0))
        self.L4  = nn.Conv2d(384*1,128,kernel_size=1,stride=1,padding=0)
        self.L3  = nn.Conv2d(192*1, 128, kernel_size=1, stride=1,padding=0)
        self.L2  = nn.Conv2d(96*1, 128, kernel_size=1, stride=1,padding=0)
        self.drop = nn.Dropout2d(p=0.0)


    def forward(self,features):
        T4 = self.L4(self.drop(features[3])).transpose(1, 3)
        T3 = self.L3(self.drop(features[2])).transpose(1, 3)
        T2 = self.L2(self.drop(features[1])).transpose(1, 3)

        S4 = torch.cat((self.deconv4(T4),T3),axis=1)
        S4 = self.conv_block4(S4)
        S43 = torch.cat((self.deconv3(S4),T2),axis=1)
        out = self.conv_block3(S43)

        return out

class Swin_RangeAngle_Decoder_RAD(nn.Module):
    def __init__(self, ):
        super(Swin_RangeAngle_Decoder_RAD, self).__init__()

        # Top-down layers
        self.deconv4 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=(2,2), padding=1, output_padding=(1,1))

        self.conv_block4 = BasicBlock(48,64)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=(2,2), padding=1, output_padding=(1,1))
        self.conv_block3 = BasicBlock(128,128)
        self.deconv2 = nn.ConvTranspose2d(256,128,kernel_size=3,stride=(1,2),padding=1,output_padding=(0,1))
        self.L4  = nn.Conv2d(384*1,128,kernel_size=1,stride=1,padding=0)
        self.L3  = nn.Conv2d(192*1, 128, kernel_size=1, stride=1,padding=0)
        self.L2  = nn.Conv2d(96*1, 128, kernel_size=1, stride=1,padding=0)
        self.drop = nn.Dropout2d(p=0.0)


    def forward(self,features):
        # Doppler, Range, Angle
        T4 = self.L4(self.drop(features[3]))
        T3 = self.L3(self.drop(features[2]))
        T2 = self.L2(self.drop(features[1]))
        S4 = torch.cat((self.deconv4(T4),T3),axis=1)
        S43 = torch.cat((self.deconv3(S4),T2),axis=1)
        S43 = self.deconv2(S43)
        out = self.conv_block3(S43)
        return out

class FFTRadNet_ViT_RAD(nn.Module):
    def __init__(self,patch_size,channels,in_chans,embed_dim,depths,num_heads,drop_rates,regression_layer = 2, detection_head=True,segmentation_head=True):
        super(FFTRadNet_ViT_RAD, self).__init__()

        self.detection_head = detection_head
        self.segmentation_head = segmentation_head

        self.vit = SwinTransformer(
                 pretrain_img_size=None,
                 patch_size=patch_size,
                 in_chans=in_chans,
                 embed_dim=embed_dim,
                 depths=depths,
                 num_heads=num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=drop_rates[0],
                 attn_drop_rate=drop_rates[1],
                 drop_path_rate=drop_rates[2],
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False)

        self.RA_decoder = Swin_RangeAngle_Decoder_RAD()

        if(self.detection_head):
            self.detection_header = Detection_Header(input_angle_size=128,reg_layer=regression_layer)

    def forward(self,x):

        out = {'Detection':[],'class':[],'Segmentation':[]}
        features = self.vit(x)
        RA = self.RA_decoder(features)
        if(self.detection_head):
            out['Detection'],out['class'] = self.detection_header(RA)

        return out

class FFTRadNet_ViT(nn.Module):
    def __init__(self,patch_size,channels,in_chans,embed_dim,depths,num_heads,drop_rates,regression_layer = 2, detection_head=True,segmentation_head=True):
        super(FFTRadNet_ViT, self).__init__()

        self.detection_head = detection_head
        self.segmentation_head = segmentation_head

        self.vit = SwinTransformer(
                 pretrain_img_size=None,
                 patch_size=patch_size,
                 in_chans=in_chans,
                 embed_dim=embed_dim,
                 depths=depths,
                 num_heads=num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=drop_rates[0],
                 attn_drop_rate=drop_rates[1],
                 drop_path_rate=drop_rates[2],
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False)

        self.RA_decoder = Swin_RangeAngle_Decoder()

        if(self.detection_head):
            self.detection_header = Detection_Header(input_angle_size=128,reg_layer=regression_layer)

    def forward(self,x):

        out = {'Detection':[],'class':[],'Segmentation':[]}
        features = self.vit(x)
        RA = self.RA_decoder(features)
        if(self.detection_head):
            out['Detection'],out['class'] = self.detection_header(RA)

        return out

class FFTRadNet_ViT_ADC(nn.Module):
    def __init__(self,patch_size,channels,in_chans,embed_dim,depths,num_heads,drop_rates,regression_layer = 2, detection_head=True,segmentation_head=True):
        super(FFTRadNet_ViT_ADC, self).__init__()

        self.detection_head = detection_head
        self.segmentation_head = segmentation_head
        self.DFT = FFT_Net()
        self.vit = SwinTransformer(
                 pretrain_img_size=None,
                 patch_size=patch_size,
                 in_chans=in_chans,
                 embed_dim=embed_dim,
                 depths=depths,
                 num_heads=num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=drop_rates[0],
                 attn_drop_rate=drop_rates[1],
                 drop_path_rate=drop_rates[2],
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False)

        self.RA_decoder = Swin_RangeAngle_Decoder()

        if(self.detection_head):
            self.detection_header = Detection_Header(input_angle_size=128,reg_layer=regression_layer)

    def forward(self,x):

        out = {'Detection':[],'class':[],'Segmentation':[]}
        x = self.DFT(x)
        features = self.vit(x)
        RA = self.RA_decoder(features)
        if(self.detection_head):
            out['Detection'],out['class'] = self.detection_header(RA)

        return out



class ADC_Net(nn.Module):
    def __init__(self,RadNet,DFT_Net):
        super(ADC_Net, self).__init__()
        self.RadNet = RadNet
        self.DFT = DFT_Net

    def forward(self,x):

        out = {'Detection':[],'class':[],'Segmentation':[]}
        x = self.DFT(x)
        x = torch.concat([x.real,x.imag],axis=1).to('cuda').float()
        out = self.RadNet(x)

        return out
