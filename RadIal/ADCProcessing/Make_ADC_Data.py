from rpl import RadarSignalProcessing
import json
import argparse
from DBReader.DBReader import SyncReader
import pandas as pd
import os

def main(config):
    cal_table = config['Calibration']
    RSP = RadarSignalProcessing(cal_table,method=config['Method'])
    labels = pd.read_csv(config['label_path'],sep=',',index_col=None)

    records = np.unique(labels['dataset'])[:1]
    data_dir = config['Data_Dir']

    for i,record in enumerate(records):
        print(i,". ",record)
        boxes = labels[labels.dataset == record]
        root_folder = os.path.join(data_dir,record)
        db = SyncReader(root_folder,tolerance=20000,silent=True)

        unique_indices = np.unique(boxes['index'])

        for index in unique_indices:

            numSample = boxes[boxes['index'] == index]['numSample'].iloc[0]
            sample = db.GetSensorData(index)

            adc=RSP.run(sample['radar_ch0']['data'],sample['radar_ch1']['data'],sample['radar_ch2']['data'],sample['radar_ch3']['data'])

            filename = os.path.join(config['Output_Folder'],"adc_{:06d}".format(numSample))
            np.save(filename,adc) 

            
if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Data Production')
    parser.add_argument('-c', '--config', default='data_config.json',type=str,
                        help='Path to the config file (default: config.json)')

    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config)