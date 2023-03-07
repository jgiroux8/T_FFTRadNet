import sys
sys.path.insert(0, '../')

from dataset.dataset import RADDet
from dataset.encoder import ra_encoder
from sklearn import preprocessing
import numpy as np


geometry = {    "ranges": [256,256,1],
                "resolution": [0.1953125,0.703125],
                "size": 3}

statistics = {  "input_mean":np.zeros(32),
                "input_std":np.ones(32),
                "reg_mean":np.zeros(3),
                "reg_std":np.ones(3)}

enc = ra_encoder(geometry = geometry,
                    statistics = statistics,
                    regression_layer = 2)

label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(["person", "bicycle", "car", "motorcycle", "bus", "truck" ])
perform_FFT = 'RD'
dataset = RADDet(root_dir = r"C:\Users\James-PC\James\RADDet",label_encoder=label_encoder,mode='Total',
                       statistics=None,
                       encoder=enc.encode,
                       difficult=True,perform_FFT=perform_FFT)

reg = []
m=0
s_real=0
s_imag=0

for i in range(len(dataset)):

    print(i,len(dataset))
    radar_FFT,out_label,box_labels,class_map = dataset.__getitem__(i)

    if perform_FFT == 'RAD':
        data = np.reshape(radar_FFT,(256*256,64))
    else:
        data = np.reshape(radar_FFT,(256*64,16))
        
        m += data.mean(axis=0)
        s_real += data.std(axis=0)



    idy,idx = np.where(out_label[0]>0)

    reg.append(out_label[1:,idy,idx])

reg = np.concatenate(reg,axis=1)

print('===  INPUT  ====')
print('mean',m/len(dataset))
print('std real',s_real/len(dataset))


print('===  Regression  ====')
print('mean',np.mean(reg,axis=1))
print('std',np.std(reg,axis=1))
