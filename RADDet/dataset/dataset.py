from torch.utils.data import Dataset
import numpy as np
import os
import torch
from torchvision.transforms import Resize,CenterCrop
import torchvision.transforms as transform
import pandas as pd
from PIL import Image
import torch.nn.functional as F
import mkl_fft
import math


class RADDet(Dataset):

    def __init__(self, root_dir,label_encoder,mode='Total',statistics=None,encoder=None,perform_FFT=False):
        #[x_center, y_center, w, h]
        self.root_dir = root_dir
        self.statistics = statistics
        self.encoder = encoder
        self.index_path = os.path.join(root_dir,'ADC')
        self.FFT_label_idx = [int(e.replace('.npy','')) for e in os.listdir(self.index_path)]
        self.test_indices = [int(e.replace('.pickle','')) for e in os.listdir(os.path.join(root_dir,'gt_box_test'))]
        self.mode = mode
        if self.mode == 'Train':
            self.FFT_label_idx = list(set(self.FFT_label_idx) - set(self.test_indices))
        elif mode == 'Test':
            self.FFT_label_idx = self.test_indices
        else:
            self.FFT_label_idx = self.FFT_label_idx

        self.numSamplePerChirp = 256
        self.numChirps = 64
        self.numRxAnt = 8
        self.numTxAnt = 2
        hanningWindowRange = (0.54 - 0.46*np.cos(((2*math.pi*np.arange(self.numSamplePerChirp ))/(self.numSamplePerChirp -1))))
        hanningWindowDoppler = (0.54 - 0.46*np.cos(((2*math.pi*np.arange(self.numChirps ))/(self.numChirps -1))))
        hanningWindowAzimuth = (0.54 - 0.46*np.cos(((2*math.pi*np.arange(self.numRxAnt ))/(self.numRxAnt -1))))
        self.range_fft_coef = np.expand_dims(np.repeat(np.expand_dims(hanningWindowRange,1), repeats=self.numChirps, axis=1),2)
        self.doppler_fft_coef = np.expand_dims(np.repeat(np.expand_dims(hanningWindowDoppler, 1).transpose(), repeats=self.numSamplePerChirp, axis=0),2)
        #self.azimuth_fft_coef = np.expand_dims(np.repeat(np.expand_dims(hanningWindowAzimuth,1).transpose(), repeats=self.numRxAnt*32, axis=0),2)
        self.label_encoder = label_encoder
        self.perform_FFT = perform_FFT

    def __len__(self):
        return len(self.FFT_label_idx)

    def __getitem__(self, index):

        # Get the sample id
        sample_id = self.FFT_label_idx[index]
        gt_name = os.path.join(self.root_dir,'gt_box','{:06d}.pickle'.format(sample_id))
        gt_box = np.load(gt_name,allow_pickle=True)
        box_labels = []

        for i in range(len(gt_box['cart_boxes'])):
            if i > 30:
                break
            # Flip the mask to have lower pixel values correspond to farther from sensor
            class_ = [gt_box['classes'][i]]
            class_ = self.label_encoder.transform(class_)[0]
            #print(class_)
            y_center = 50 - gt_box['cart_boxes'][i][0]*50/256
            # Cartesian Grid Spans -50 to 50, split pixels in half.
            x_center = gt_box['cart_boxes'][i][1]*100/512 - 50.
            R = np.sqrt(x_center**2 + y_center**2)
            phi = np.degrees(np.arctan(x_center/y_center))
            box_labels.append([R,phi,class_])

        ######################
        #  Encode the labels #
        ######################
        out_label=[]
        if(self.encoder!=None):
            out_label = self.encoder(box_labels).copy()
            class_map = out_label[3:]
            out_label = out_label[:3]


        # Read the Radar ADC data

        radar_name = os.path.join(self.root_dir,'ADC',"{:06d}.npy".format(sample_id))
        complex_adc = np.load(radar_name).reshape(256,64,8)

        if self.perform_FFT == 'ADC':
            radar_FFT = complex_adc

        elif self.perform_FFT == 'RD':
            complex_adc = complex_adc - np.mean(complex_adc, axis=(0,1))
            # 3- Range FFTs
            # No Windowing
            #range_fft = mkl_fft.fft(complex_adc,self.numSamplePerChirp,axis=0)
            # Windowing
            range_fft = mkl_fft.fft(np.multiply(complex_adc,self.range_fft_coef),self.numSamplePerChirp,axis=0)
            # DONT Shift range zero frequency to center of spectrum
            # 4- Doppler FFts
            # Windowing
            input = mkl_fft.fft(np.multiply(range_fft,self.doppler_fft_coef),self.numChirps,axis=1)
            # No Windowing
            # input = mkl_fft.fft(range_fft,self.numChirps,axis=1)
            # Shift doppler zero freqency bin to center of spectrum
            input = np.fft.fftshift(input,axes=1)
            radar_FFT = np.concatenate([input.real,input.imag],axis=2)

            if(self.statistics is not None):
                for i in range(len(self.statistics['input_mean'])):
                    radar_FFT[...,i] -= self.statistics['input_mean'][i]
                    radar_FFT[...,i] /= self.statistics['input_std'][i]

        elif self.perform_FFT == 'RAD':
            # 3- Range FFTs
            range_fft = mkl_fft.fft(np.multiply(complex_adc,self.range_fft_coef),self.numSamplePerChirp,axis=0)
            # DONT Shift range zero frequency to center of spectrum
            # 4- Doppler FFts
            input = mkl_fft.fft(np.multiply(range_fft,self.doppler_fft_coef),self.numChirps,axis=1)
            # Shift doppler zero freqency bin to center of spectrum
            RD = np.fft.fftshift(input,axes=1)
            #RD = np.concatenate([input.real,input.imag],axis=2)

            zeros = np.zeros((256,64,256-8)).astype('complex64')
            RAD = np.concatenate([RD,zeros],axis=2)
            RAD = mkl_fft.fft(RAD,axis=2)
            RAD = np.fft.fftshift(RAD,axes=2)
            radar_FFT = np.abs(np.transpose(RAD,axes=[0,2,1]))

            if(self.statistics is not None):
                for i in range(len(self.statistics['input_mean'])):
                    radar_FFT[...,i] -= self.statistics['input_mean'][i]
                    radar_FFT[...,i] /= self.statistics['input_std'][i]

        return radar_FFT,out_label,np.asarray(box_labels),class_map
