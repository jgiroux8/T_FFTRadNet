import os
import json
import argparse
import torch
import random
import numpy as np
from model.FFTRadNet_ViT import FFTRadNet_ViT_ADC,FFTRadNet_ViT_RAD,FFTRadNet_ViT
from dataset.dataset import RADDet
from dataset.encoder import ra_encoder
from dataset.dataloader import CreateDataLoaders
import pkbar
import torch.nn.functional as F
from utils.evaluation import run_FullEvaluation
import torch.nn as nn
from utils.metrics import run_APEvaluation
from sklearn import preprocessing


def main(config, checkpoint):

    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(["person", "bicycle", "car", "motorcycle", "bus", "truck" ])
    # Load the dataset
    enc = ra_encoder(geometry = config['dataset']['geometry'],
                        statistics = config['dataset']['statistics'],
                        regression_layer = 2)

    dataset = RADDet(root_dir = config['dataset']['root_dir'],label_encoder=label_encoder,mode='Train',
                        statistics= config['dataset']['statistics'],
                        encoder=enc.encode,perform_FFT=config['data_mode'])

    test_dataset = RADDet(root_dir = config['dataset']['root_dir'],label_encoder=label_encoder,mode='Test',
                       statistics=config['dataset']['statistics'],
                       encoder=enc.encode,perform_FFT=config['data_mode'])

    train_loader, val_loader, test_loader = CreateDataLoaders(dataset,test_dataset,config['dataloader'],config['seed'])



    # Create the model
    if config['data_mode'] == 'RAD':
        net = FFTRadNet_ViT_RAD(patch_size = config['model']['patch_size'],
                        channels = config['model']['channels'],
                        in_chans = config['model']['in_chans'],
                        embed_dim = config['model']['embed_dim'],
                        depths = config['model']['depths'],
                        num_heads = config['model']['num_heads'],
                        drop_rates = config['model']['drop_rates'],
                        regression_layer = 2,
                        detection_head = config['model']['DetectionHead'],
                        segmentation_head = config['model']['SegmentationHead'])

    elif config['data_mode'] == 'RD':
        net = FFTRadNet_ViT(patch_size = config['model']['patch_size'],
                        channels = config['model']['channels'],
                        in_chans = config['model']['in_chans'],
                        embed_dim = config['model']['embed_dim'],
                        depths = config['model']['depths'],
                        num_heads = config['model']['num_heads'],
                        drop_rates = config['model']['drop_rates'],
                        regression_layer = 2,
                        detection_head = config['model']['DetectionHead'],
                        segmentation_head = config['model']['SegmentationHead'])

    else:
        net = FFTRadNet_ViT_ADC(patch_size = config['model']['patch_size'],
                        channels = config['model']['channels'],
                        in_chans = config['model']['in_chans'],
                        embed_dim = config['model']['embed_dim'],
                        depths = config['model']['depths'],
                        num_heads = config['model']['num_heads'],
                        drop_rates = config['model']['drop_rates'],
                        regression_layer = 2,
                        detection_head = config['model']['DetectionHead'],
                        segmentation_head = config['model']['SegmentationHead'])

    print("Parameters: ",sum(p.numel() for p in net.parameters() if p.requires_grad))

    net.to('cuda')


    print('===========  Loading the model ==================:')
    dict = torch.load(checkpoint)
    net.load_state_dict(dict['net_state_dict'])

    print('===========  Running the evaluation ==================:')
    run_APEvaluation(net,test_loader,enc,iou_threshold=0.5,config=config)
    run_FullEvaluation(net,test_loader,enc,config=config)





if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FFTRadNet Evaluation')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--checkpoint', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    #parser.add_argument('--difficult', action='store_true')
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config, args.checkpoint)
