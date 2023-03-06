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

class RADIal(Dataset):

    def __init__(self, root_dir,statistics=None,encoder=None,difficult=False,perform_FFT='Default'):

        self.root_dir = root_dir
        self.statistics = statistics
        self.encoder = encoder
        self.perform_FFT = perform_FFT
        self.labels = pd.read_csv(os.path.join(root_dir,'labels.csv')).to_numpy()
        self.numChirps = 256
        self.numSamplePerChirp = 512
        self.numRxAnt = 16
        self.numTxAnt = 12
        hanningWindowRange = (0.54 - 0.46*np.cos(((2*math.pi*np.arange(self.numSamplePerChirp ))/(self.numSamplePerChirp -1))))
        hanningWindowDoppler = (0.54 - 0.46*np.cos(((2*math.pi*np.arange(self.numChirps ))/(self.numChirps -1))))
        hanningWindowAzimuth = (0.54 - 0.46*np.cos(((2*math.pi*np.arange(self.numRxAnt ))/(self.numRxAnt -1))))
        self.range_fft_coef = np.expand_dims(np.repeat(np.expand_dims(hanningWindowRange,1), repeats=self.numChirps, axis=1),2)
        self.doppler_fft_coef = np.expand_dims(np.repeat(np.expand_dims(hanningWindowDoppler, 1).transpose(), repeats=self.numSamplePerChirp, axis=0),2)

        # Keeps only easy samples
        if(difficult==False):
            ids_filters=[]
            ids = np.where( self.labels[:, -1] == 0)[0]
            ids_filters.append(ids)
            ids_filters = np.unique(np.concatenate(ids_filters))
            self.labels = self.labels[ids_filters]


        # Gather each input entries by their sample id
        self.unique_ids = np.unique(self.labels[:,0])
        self.label_dict = {}
        for i,ids in enumerate(self.unique_ids):
            sample_ids = np.where(self.labels[:,0]==ids)[0]
            self.label_dict[ids]=sample_ids
        self.sample_keys = list(self.label_dict.keys())


        self.resize = Resize((256,224), interpolation=transform.InterpolationMode.NEAREST)
        self.crop = CenterCrop((512,448))


    def __len__(self):
        return len(self.label_dict)

    def __getitem__(self, index):

        # Get the sample id
        sample_id = self.sample_keys[index]

        # From the sample id, retrieve all the labels ids
        entries_indexes = self.label_dict[sample_id]

        # Get the objects labels
        box_labels = self.labels[entries_indexes]

        # Labels contains following parameters:
        # x1_pix	y1_pix	x2_pix	y2_pix	laser_X_m	laser_Y_m	laser_Z_m radar_X_m	radar_Y_m	radar_R_m

        # format as following [Range, Angle, Doppler,laser_X_m,laser_Y_m,laser_Z_m,x1_pix,y1_pix,x2_pix	,y2_pix]
        box_labels = box_labels[:,[10,11,12,5,6,7,1,2,3,4]].astype(np.float32)


        ######################
        #  Encode the labels #
        ######################
        out_label=[]
        if(self.encoder!=None):
            out_label = self.encoder(box_labels).copy()

        # Read the Radar data
        if self.perform_FFT == 'Custom_RD':
            # If you want to perform custom operations on the radar.
            # Here we include an fftshift to center zero doppler frequency
            # For a rectangular window remove multiplication of coefs
            # I recommend saving them with the commented code at the bottom
            # of the if statement and running 'dataset/print_dataset_statistcs.py'
            # This will give you normalization values and save to the folder you have created under the root_dir
            radar_name = os.path.join(self.root_dir,'ADC_Data',"adc_{:06d}.npy".format(sample_id))
            complex_adc = np.load(radar_name,allow_pickle=True)
            complex_adc = complex_adc - np.mean(complex_adc, axis=(0,1))
            range_fft = mkl_fft.fft(np.multiply(complex_adc,self.range_fft_coef),self.numSamplePerChirp,axis=0)
            input = mkl_fft.fft(np.multiply(range_fft,self.doppler_fft_coef),self.numChirps,axis=1)
            # Shift doppler zero freqency bin to center of spectrum
            radar_FFT = np.fft.fftshift(input,axes=1).astype(np.complex64)
            radar_FFT = np.concatenate([input.real,input.imag],axis=2)

            if(self.statistics is not None):
                for i in range(len(self.statistics['input_mean'])):
                    radar_FFT[...,i] -= self.statistics['input_mean'][i]
                    radar_FFT[...,i] /= self.statistics['input_std'][i]

            #out_name = os.path.join(self.root_dir,'RD_Shift',"fft_{:06d}.npy".format(sample_id))
            #np.save(out_name,radar_FFT)

        elif self.perform_FFT == 'ADC':
            # Utilize Raw ADC
            radar_name = os.path.join(self.root_dir,'ADC_Data',"adc_{:06d}.npy".format(sample_id))
            radar_FFT = np.load(radar_name,allow_pickle=True)

        else:
            # Default method. Uses precomputed FFT's.
            radar_name = os.path.join(self.root_dir,'radar_FFT',"fft_{:06d}.npy".format(sample_id))
            input = np.load(radar_name,allow_pickle=True)
            radar_FFT = np.concatenate([input.real,input.imag],axis=2)
            if(self.statistics is not None):
                for i in range(len(self.statistics['input_mean'])):
                    radar_FFT[...,i] -= self.statistics['input_mean'][i]
                    radar_FFT[...,i] /= self.statistics['input_std'][i]

        # Read the segmentation map
        segmap_name = os.path.join(self.root_dir,'radar_Freespace',"freespace_{:06d}.png".format(sample_id))
        segmap = Image.open(segmap_name) # [512,900]
        # 512 pix for the range and 900 pix for the horizontal FOV (180deg)
        # We crop the fov to 89.6deg
        segmap = self.crop(segmap)
        # and we resize to half of its size
        segmap = np.asarray(self.resize(segmap))==255

        # Read the camera image
        img_name = os.path.join(self.root_dir,'camera',"image_{:06d}.jpg".format(sample_id))
        image = np.asarray(Image.open(img_name))

        return radar_FFT, segmap,out_label,box_labels,image
