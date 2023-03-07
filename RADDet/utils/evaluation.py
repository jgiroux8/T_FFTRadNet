import torch
import numpy as np
from .metrics import GetFullMetrics, Metrics
import pkbar
import pickle
import torch.nn as nn

def run_evaluation(net,loader,encoder,check_perf=False, detection_loss=None,segmentation_loss=None,losses_params=None,config=None):

    metrics = Metrics()
    metrics.reset()

    net.eval()
    running_loss = 0.0
    class_loss = nn.CrossEntropyLoss(reduction='mean')
    kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False)

    for i, data in enumerate(loader):

        # input, out_label,segmap,labels
        if config['data_mode'] == 'ADC':
            inputs = data[0].to('cuda').type(torch.complex64)
        else:
            inputs = data[0].to('cuda').float()

        label_map = data[1].to('cuda').float()
        class_map = data[3].to('cuda').float()

        with torch.set_grad_enabled(False):
            outputs = net(inputs)

        if(detection_loss!=None):
            classif_loss,reg_loss = detection_loss(outputs['Detection'], label_map,losses_params)
            obj_loss = class_loss(outputs['class'],class_map)

            classif_loss *= losses_params['weight'][0]
            reg_loss *= losses_params['weight'][1]
            obj_loss *= losses_params['weight'][2]


            loss = classif_loss + reg_loss + obj_loss

            # statistics
            running_loss += loss.item() * inputs.size(0)

        if(check_perf):
            out_obj = outputs['Detection'].detach().cpu().numpy().copy()
            labels = data[2]
            out_class = outputs['class'].detach().cpu().numpy().copy()

            out_obj = np.concatenate([out_obj,out_class],axis=1)

            for pred_obj,true_obj in zip(out_obj,labels):

                metrics.update(np.asarray(encoder.decode(pred_obj,0.05)),true_obj,
                            threshold=0.2,range_min=0,range_max=50)



        kbar.update(i)


    mAP,mAR = metrics.GetMetrics()

    return {'loss':running_loss / len(loader.dataset) , 'mAP':mAP, 'mAR':mAR}


def run_FullEvaluation(net,loader,encoder,iou_threshold=0.5,config=None):

    net.eval()
    results = []
    kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False)

    print('Generating Predictions...')
    predictions = {'prediction':{'objects':[],'freespace':[]},'label':{'objects':[],'freespace':[]}}
    for i, data in enumerate(loader):

        if config['data_mode'] == 'ADC':
            inputs = data[0].to('cuda').type(torch.complex64)
        else:
            inputs = data[0].to('cuda').float()

        with torch.set_grad_enabled(False):
            outputs = net(inputs)

        out_obj = outputs['Detection'].detach().cpu().numpy().copy()
        out_class = outputs['class'].detach().cpu().numpy().copy()

        labels_object = data[2]
        out_obj = np.concatenate([out_obj,out_class],axis=1)

        for pred_obj,true_obj in zip(out_obj,labels_object):

            predictions['prediction']['objects'].append( np.asarray(encoder.decode(pred_obj,0.05)))
            predictions['label']['objects'].append(true_obj)



        kbar.update(i)

    iou_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    for iou_ in iou_list:
        results.append(GetFullMetrics(predictions['prediction']['objects'],predictions['label']['objects'],range_min=0,range_max=50,IOU_threshold=iou_))
