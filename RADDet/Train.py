import os
import json
import argparse
import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from model.FFTRadNet_ViT import FFTRadNet_ViT,FFTRadNet_ViT_ADC,FFTRadNet_ViT_RAD
from dataset.dataset import RADDet
from dataset.encoder import ra_encoder
from dataset.dataloader import CreateDataLoaders
import pkbar
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from loss import pixor_loss
from utils.evaluation import run_evaluation
import torch.nn as nn
from functools import partial
from utils.metrics import run_APEvaluation
from sklearn import preprocessing


def main(config, resume):

    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    # create experience name
    curr_date = datetime.now()
    exp_name = config['name'] + '___' + curr_date.strftime('%b-%d-%Y___%H:%M:%S')
    exp_name = exp_name[:-11]
    print(exp_name)

    # Create directory structure
    output_folder = config['output']['dir']#Path(config['output']['dir'])
    #output_folder.mkdir(parents=True, exist_ok=True)
    #print(os.path.join(output_folder,exp_name))
    #print("********")
    os.mkdir(os.path.join(output_folder,exp_name))
    #(output_folder / exp_name).mkdir(parents=True, exist_ok=True)
    # and copy the config file
    #with open(output_folder / exp_name / 'config.json', 'w') as outfile:
    with open(os.path.join(output_folder,exp_name,'config.json'),'w') as outfile:
        json.dump(config, outfile)

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize tensorboard
    writer = SummaryWriter(os.path.join(output_folder, exp_name))
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(["person", "bicycle", "car", "motorcycle", "bus", "truck" ])
    # Load the dataset
    enc = ra_encoder(geometry = config['dataset']['geometry'],
                        statistics = config['dataset']['statistics'],
                        regression_layer = 2)

    dataset = RADDet(root_dir = config['dataset']['root_dir'],label_encoder=label_encoder,mode='Train',
                        statistics=config['dataset']['statistics'],
                        encoder=enc.encode,perform_FFT=config['data_mode'])

    test_dataset = RADDet(root_dir = config['dataset']['root_dir'],label_encoder=label_encoder,mode='Test',
                       statistics= config['dataset']['statistics'],
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



    t_params = sum(p.numel() for p in net.parameters())
    print("Network Parameters: ",t_params)
    net.to('cuda')
    print(net)

    # Optimizer
    lr = float(config['optimizer']['lr'])
    step_size = int(config['lr_scheduler']['step_size'])
    gamma = float(config['lr_scheduler']['gamma'])
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    class_loss = nn.CrossEntropyLoss(reduction='mean')
    num_epochs=int(config['num_epochs'])


    print('===========  Optimizer  ==================:')
    print('      LR:', lr)
    print('      step_size:', step_size)
    print('      gamma:', gamma)
    print('      num_epochs:', num_epochs)
    print('')

    # Train
    startEpoch = 0
    global_step = 0
    history = {'train_loss':[],'val_loss':[],'lr':[],'mAP':[],'mAR':[],'mIoU':[]}
    best_mAP = 0

    freespace_loss = nn.BCEWithLogitsLoss(reduction='mean')


    if resume:
        print('===========  Resume training  ==================:')
        dict = torch.load(resume)
        net.load_state_dict(dict['net_state_dict'])
        optimizer.load_state_dict(dict['optimizer'])
        scheduler.load_state_dict(dict['scheduler'])
        startEpoch = dict['epoch']+1
        history = dict['history']
        global_step = dict['global_step']

        print('       ... Start at epoch:',startEpoch)



    for epoch in range(startEpoch,num_epochs):

        kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)

        ###################
        ## Training loop ##
        ###################
        net.train()
        running_loss = 0.0


        for i, data in enumerate(train_loader):
            #if i == 10:
            #    break
            if config['data_mode'] == 'ADC':
                inputs = data[0].to('cuda').type(torch.complex64)
            else:
                inputs = data[0].to('cuda').float()

            label_map = data[1].to('cuda').float()
            class_map = data[3].to('cuda').float()
            if(config['model']['SegmentationHead']=='True'):
                seg_map_label = data[2].to('cuda').double()

            # reset the gradient
            optimizer.zero_grad()

            # forward pass, enable to track our gradient
            with torch.set_grad_enabled(True):
                outputs = net(inputs)

            classif_loss,reg_loss = pixor_loss(outputs['Detection'], label_map,config['losses'])
            obj_loss = class_loss(outputs['class'],class_map)


            classif_loss *= config['losses']['weight'][0]
            reg_loss *= config['losses']['weight'][1]
            obj_loss *= config['losses']['weight'][2]


            loss = classif_loss + reg_loss + obj_loss

            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Loss/train_clc', classif_loss.item(), global_step)
            writer.add_scalar('Loss/train_reg', reg_loss.item(), global_step)
            writer.add_scalar('Loss/class',obj_loss.item(),global_step)

            # backprop
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)

            kbar.update(i, values=[("loss", loss.item()), ("class", classif_loss.item()), ("reg", reg_loss.item()),("obj", obj_loss.item())])#,("freeSpace", loss_seg.item())])


            global_step += 1


        scheduler.step()

        history['train_loss'].append(running_loss / len(train_loader.dataset))
        history['lr'].append(scheduler.get_last_lr()[0])


        ######################
        ## validation phase ##
        ######################

        eval = run_evaluation(net,val_loader,enc,check_perf=(epoch>=10),
                                detection_loss=pixor_loss,segmentation_loss=None,
                                losses_params=config['losses'],config=config)

        if epoch >= 20 and epoch % 2 == 0:
            run_APEvaluation(net,val_loader,enc,iou_threshold=0.5,config=config)

        history['val_loss'].append(eval['loss'])
        history['mAP'].append(eval['mAP'])
        history['mAR'].append(eval['mAR'])

        if eval['mAP'] + eval['mAR'] == 0:
            F1_score = 0.0
        else:
            F1_score = (eval['mAP']*eval['mAR'])/((eval['mAP'] + eval['mAR'])/2)
        kbar.add(1, values=[("val_loss", eval['loss']),("mAP", eval['mAP']),("mAR", eval['mAR']),('F1',F1_score)])


        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
        writer.add_scalar('Loss/test', eval['loss'], global_step)
        writer.add_scalar('Metrics/mAP', eval['mAP'], global_step)
        writer.add_scalar('Metrics/mAR', eval['mAR'], global_step)

        # Saving all checkpoint as the best checkpoint for multi-task is a balance between both --> up to the user to decide
        name_output_file = config['name']+'_epoch{:02d}_loss_{:.4f}_AP_{:.4f}_AR_{:.4f}_F1_{:.4f}.pth'.format(epoch, eval['loss'],eval['mAP'],eval['mAR'],F1_score)
        filename = os.path.join(output_folder , exp_name , name_output_file)

        checkpoint={}
        checkpoint['net_state_dict'] = net.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint['scheduler'] = scheduler.state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['history'] = history
        checkpoint['global_step'] = global_step

        torch.save(checkpoint,filename)

        print('')




if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FFTRadNet Training')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')

    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config, args.resume)
