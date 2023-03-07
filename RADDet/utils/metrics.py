from typing import Final
import torch
import os
import json
import numpy as np
import pkbar
import argparse
from shapely.geometry import Polygon
from shapely.ops import unary_union
from sklearn import preprocessing
import pickle

def convert_RA_box(data):
    L = 2
    W = 2
    boxes = []
    for i in range(len(data)):
        R = data[i][0]
        A = data[i][1]
        boxes.append([R-L/2,A-W/2,R+L/2,A-W/2,R+L/2,A+L/2,R-L/2,A+W/2])

    return boxes

def RA_to_cartesian_box(data):
    L = 4
    W = 1.8

    boxes = []
    for i in range(len(data)):

        x = np.sin(np.radians(data[i][1])) * data[i][0]
        y = np.cos(np.radians(data[i][1])) * data[i][0]

        boxes.append([x - W/2,y,x + W/2,y, x + W/2,y+L,x - W/2,y+L])#,data[i][0],data[i][1]])

    return boxes

def perform_nms(valid_class_predictions, valid_box_predictions, nms_threshold):

    # sort the detections such that the entry with the maximum confidence score is at the top
    sorted_indices = np.argsort(valid_class_predictions)[::-1]
    sorted_box_predictions = valid_box_predictions[sorted_indices]
    sorted_class_predictions = valid_class_predictions[sorted_indices]

    for i in range(sorted_box_predictions.shape[0]):
        # get the IOUs of all boxes with the currently most certain bounding box
        try:
            ious = np.zeros((sorted_box_predictions.shape[0]))
            ious[i + 1:] = bbox_iou(sorted_box_predictions[i, :], sorted_box_predictions[i + 1:, :])
        except ValueError:
            break
        except IndexError:
            break

        # eliminate all detections which have IoU > threshold
        overlap_mask = np.where(ious < nms_threshold, True, False)
        sorted_box_predictions = sorted_box_predictions[overlap_mask]
        sorted_class_predictions = sorted_class_predictions[overlap_mask]

    return sorted_class_predictions, sorted_box_predictions
def bbox_iou(box1, boxes):

    # currently inspected box
    box1 = box1.reshape((4,2))
    rect_1 = Polygon([(box1[0, 0], box1[0, 1]), (box1[1, 0], box1[1, 1]), (box1[2, 0], box1[2, 1]),
                      (box1[3, 0], box1[3, 1])])
    area_1 = rect_1.area

    # IoU of box1 with each of the boxes in "boxes"
    ious = np.zeros(boxes.shape[0])
    for box_id in range(boxes.shape[0]):
        box2 = boxes[box_id]
        box2 = box2.reshape((4,2))
        rect_2 = Polygon([(box2[0, 0], box2[0, 1]), (box2[1, 0], box2[1, 1]), (box2[2, 0], box2[2, 1]),
                          (box2[3, 0], box2[3, 1])])
        area_2 = rect_2.area

        # get intersection of both bounding boxes
        inter_area = rect_1.intersection(rect_2).area

        # compute IoU of the two bounding boxes
        iou = inter_area / (area_1 + area_2 - inter_area)

        ious[box_id] = iou

    return ious

def process_predictions_FFT(batch_predictions, confidence_threshold=0.1, nms_threshold=0.05):

    # process targets and perform NMS for each prediction in batch
    final_batch_predictions = None  # store final bounding box predictions

    point_cloud_reg_predictions = RA_to_cartesian_box(batch_predictions)
    point_cloud_reg_predictions = np.asarray(point_cloud_reg_predictions)
    point_cloud_class_predictions = batch_predictions[:,2]

    # get valid detections
    validity_mask = np.where(point_cloud_class_predictions > confidence_threshold, True, False)

    valid_box_predictions = point_cloud_reg_predictions[validity_mask]
    valid_class_predictions = point_cloud_class_predictions[validity_mask]


    # perform Non-Maximum Suppression
    final_class_predictions, final_box_predictions = perform_nms(valid_class_predictions, valid_box_predictions,
                                                                 nms_threshold)


    # concatenate point_cloud_id, confidence score and bounding box prediction | shape: [N_FINAL, 1+1+8]
    final_Object_predictions = np.hstack((final_class_predictions[:, np.newaxis],
                                               final_box_predictions))


    return final_Object_predictions

def GetFullMetrics(predictions,object_labels,range_min=5,range_max=100,IOU_threshold=0.5):
    perfs = {}
    precision = []
    recall = []
    iou_threshold = []
    RangeError = []
    AngleError = []

    out = []
    #thresh_list = [0.3]
    for threshold in np.arange(0.1,0.96,0.1):
    #for threshold in thresh_list:
        iou_threshold.append(threshold)

        TP = 0
        FP = 0
        FN = 0
        NbDet = 0
        NbGT = 0
        NBFrame = 0
        range_error=0
        angle_error=0
        nbObjects = 0

        for frame_id in range(len(predictions)):

            pred= predictions[frame_id]
            labels = object_labels[frame_id]

            # get final bounding box predictions
            Object_predictions = []
            ground_truth_box_corners = []

            if(len(pred)>0):
                Object_predictions = process_predictions_FFT(pred,confidence_threshold=threshold,nms_threshold=0.3)

            if(len(Object_predictions)>0):
                max_distance_predictions = (Object_predictions[:,2]+Object_predictions[:,4])/2
                ids = np.where((max_distance_predictions>=range_min) & (max_distance_predictions <= range_max) )
                Object_predictions = Object_predictions[ids]

            NbDet += len(Object_predictions)

            if(len(labels)>0):
                ids = np.where((labels[:,0]>=range_min) & (labels[:,0] <= range_max))
                labels = labels[ids]

            if(len(labels)>0):
                ground_truth_box_corners = np.asarray(RA_to_cartesian_box(labels))
                NbGT += ground_truth_box_corners.shape[0]

            # valid predictions and labels exist for the currently inspected point cloud
            if len(ground_truth_box_corners)>0 and len(Object_predictions)>0:

                used_gt = np.zeros(len(ground_truth_box_corners))
                for pid, prediction in enumerate(Object_predictions):
                    iou = bbox_iou(prediction[1:], ground_truth_box_corners)
                    ids = np.where(iou>=IOU_threshold)[0]


                    if(len(ids)>0):
                        TP += 1
                        used_gt[ids]=1

                        # cummulate errors
                        range_error += np.sum(np.abs(ground_truth_box_corners[ids,-2] - prediction[-2]))
                        angle_error += np.sum(np.abs(ground_truth_box_corners[ids,-1] - prediction[-1]))
                        nbObjects+=len(ids)
                    else:
                        FP+=1
                FN += np.sum(used_gt==0)


            elif(len(ground_truth_box_corners)==0):
                FP += len(Object_predictions)
            elif(len(Object_predictions)==0):
                FN += len(ground_truth_box_corners)



        if(TP!=0):
            precision.append( TP / (TP+FP)) # When there is a detection, how much I m sure
            recall.append(TP / (TP+FN))
        else:
            precision.append( 0) # When there is a detection, how much I m sure
            recall.append(0)

        if nbObjects > 0:
            RangeError.append(range_error/nbObjects)
            AngleError.append(angle_error/nbObjects)

    perfs['precision']=precision
    perfs['recall']=recall

    F1_score = (np.mean(precision)*np.mean(recall))/((np.mean(precision) + np.mean(recall))/2)

    print('------- Detection Scores - IOU Threshold {0} ------------'.format(IOU_threshold))
    print('  mAP:',np.mean(perfs['precision']))
    print('  mAR:',np.mean(perfs['recall']))
    print('  F1 score:',F1_score)

    print('------- Regression Errors------------')
    print('  Range Error:',np.mean(RangeError),'m')
    print('  Angle Error:',np.mean(AngleError),'degree')

    return [IOU_threshold,np.mean(perfs['precision']),np.mean(perfs['recall']),F1_score,np.mean(RangeError),np.mean(AngleError)]

def GetDetMetrics(predictions,object_labels,threshold=0.2,range_min=0,range_max=50,IOU_threshold=0.5):

    TP = 0
    FP = 0
    FN = 0
    NbDet=0
    NbGT=0

    # get final bounding box predictions
    Object_predictions = []
    ground_truth_box_corners = []
    labels=[]

    if(len(predictions)>0):
        Object_predictions = process_predictions_FFT(predictions,confidence_threshold=threshold,nms_threshold=0.3)

    if(len(Object_predictions)>0):
        max_distance_predictions = (Object_predictions[:,2]+Object_predictions[:,4])/2
        ids = np.where((max_distance_predictions>=range_min) & (max_distance_predictions <= range_max) )
        Object_predictions = Object_predictions[ids]

    NbDet = len(Object_predictions)

    if(len(object_labels)>0):
        ids = np.where((object_labels[:,0]>=range_min) & (object_labels[:,0] <= range_max))
        labels = object_labels[ids]
    if(len(labels)>0):
        ground_truth_box_corners = np.asarray(RA_to_cartesian_box(labels))
        NbGT = len(ground_truth_box_corners)

    # valid predictions and labels exist for the currently inspected point cloud
    if NbDet>0 and NbGT>0:

        used_gt = np.zeros(len(ground_truth_box_corners))

        for pid, prediction in enumerate(Object_predictions):
            iou = bbox_iou(prediction[1:], ground_truth_box_corners)
            ids = np.where(iou>=IOU_threshold)[0]

            if(len(ids)>0):
                TP += 1
                used_gt[ids]=1
            else:
                FP+=1
        FN += np.sum(used_gt==0)

    elif(NbGT==0):
        FP += NbDet
    elif(NbDet==0):
        FN += NbGT

    return TP,FP,FN


class Metrics():
    def __init__(self,):

        self.iou = []
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.precision = 0
        self.recall = 0
        self.mIoU =0

    def update(self,ObjectPred,Objectlabels,threshold=0.2,range_min=0,range_max=50):
        TP,FP,FN = GetDetMetrics(ObjectPred,Objectlabels,threshold=0.2,range_min=range_min,range_max=range_max)

        self.TP += TP
        self.FP += FP
        self.FN += FN

    def reset(self,):
        self.iou = []
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.precision = 0
        self.mIoU =0

    def GetMetrics(self,):

        if(self.TP+self.FP!=0):
            self.precision = self.TP / (self.TP+self.FP)
        if(self.TP+self.FN!=0):
            self.recall = self.TP / (self.TP+self.FN)

        if(len(self.iou)>0):
            self.mIoU = np.asarray(self.iou).mean()

        return self.precision,self.recall#,self.mIoU


def run_APEvaluation(net,loader,encoder,iou_threshold=0.5,config=None):
    soft = torch.nn.Softmax(dim=1)
    net.eval()
    results = []
    kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False)
    print(" ")
    print('Generating Predictions...')
    predictions = {'prediction':{'objects':[],'classes':[]},'label':{'objects':[],'classes':[]}}
    for i, data in enumerate(loader):
        # input, out_label,segmap,labels
        if config['data_mode'] == 'ADC':
            inputs = data[0].to('cuda').type(torch.complex64)
        else:
            inputs = data[0].to('cuda').float()

        with torch.set_grad_enabled(False):
            outputs = net(inputs)

        out_obj = outputs['Detection'].detach().cpu().numpy().copy()
        out_class = soft(outputs['class']).detach().cpu().numpy().copy()
        out_obj = np.concatenate([out_obj,out_class],axis=1)
        labels_object = data[2]

        for pred_obj,true_obj in zip(out_obj,labels_object):

            predictions['prediction']['objects'].append( np.asarray(encoder.decode(pred_obj,0.05)))
            predictions['label']['objects'].append(np.asarray(true_obj.detach().cpu().numpy().copy()))

        kbar.update(i)

    iou_ = 0.5
    GetAPMetrics(predictions['prediction']['objects'],predictions['label']['objects'],range_min=0,range_max=50,IOU_threshold=iou_)


def GetAPMetrics(predictions,object_labels,range_min=5,range_max=100,IOU_threshold=0.5):
    perfs = {}
    iou_threshold = []
    RangeError = []
    AngleError = []
    all_p = []
    all_r = []
    label_encoder = preprocessing.LabelEncoder()
    label_encoder = label_encoder.fit(["person", "bicycle", "car", "motorcycle", "bus", "truck" ])

    out = []
    threshold = 0.2
    motorcycles = []
    alls = []
    results_AP = {'bicycle':[],'bus':[],'car':[],'motorcycle':[],'person':[],'truck':[],'mAP':[],'mAR':[]}
    for class_ in [0,1,2,3,4,5]:
        for threshold in np.arange(0.1,.96,0.05):
            TP = 0
            FP = 0
            FN = 0
            NbDet = 0
            NbGT = 0
            NBFrame = 0
            range_error=0
            angle_error=0
            nbObjects = 0
            precision = []
            recall = []
            for frame_id in range(len(predictions)):

                pred= predictions[frame_id]
                labels = object_labels[frame_id]
                if len(pred > 0):
                    pred = pred[np.where(pred[:,3] == class_)]

                if len(labels > 0):
                    labels = labels[np.where(labels[:,2] == class_)]

                # get final bounding box predictions
                Object_predictions = []
                ground_truth_box_corners = []

                if(len(pred)>0):
                    Object_predictions = process_predictions_FFT(pred,confidence_threshold=threshold,nms_threshold=0.3)

                if(len(Object_predictions)>0):
                    max_distance_predictions = (Object_predictions[:,2]+Object_predictions[:,4])/2
                    ids = np.where((max_distance_predictions>=range_min) & (max_distance_predictions <= range_max) )
                    Object_predictions = Object_predictions[ids]

                NbDet += len(Object_predictions)

                if(len(labels)>0):
                    ids = np.where((labels[:,0]>=range_min) & (labels[:,0] <= range_max))
                    labels = labels[ids]

                if(len(labels)>0):
                    ground_truth_box_corners = np.asarray(RA_to_cartesian_box(labels))
                    NbGT += ground_truth_box_corners.shape[0]

                # valid predictions and labels exist for the currently inspected point cloud
                if len(ground_truth_box_corners)>0 and len(Object_predictions)>0:

                    used_gt = np.zeros(len(ground_truth_box_corners))
                    for pid, prediction in enumerate(Object_predictions):
                        iou = bbox_iou(prediction[1:], ground_truth_box_corners)
                        ids = np.where(iou>=IOU_threshold)[0]


                        if(len(ids)>0):
                            TP += 1
                            used_gt[ids]=1

                            # cummulate errors
                            range_error += np.sum(np.abs(ground_truth_box_corners[ids,-2] - prediction[-2]))
                            angle_error += np.sum(np.abs(ground_truth_box_corners[ids,-1] - prediction[-1]))
                            nbObjects+=len(ids)
                        else:
                            FP+=1
                    FN += np.sum(used_gt==0)


                elif(len(ground_truth_box_corners)==0):
                    FP += len(Object_predictions)
                elif(len(Object_predictions)==0):
                    FN += len(ground_truth_box_corners)



            if(TP!=0):
                precision.append( TP / (TP+FP)) # When there is a detection, how much I m sure
                recall.append(TP / (TP+FN))
            else:
                precision.append( 0) # When there is a detection, how much I m sure
                recall.append(0)

            if nbObjects > 0:
                RangeError.append(range_error/nbObjects)
                AngleError.append(angle_error/nbObjects)

        F1_score = (np.mean(precision)*np.mean(recall))/((np.mean(precision) + np.mean(recall) + 1e-6)/2)
        all_p.append(np.sum(precision)/len(precision))
        all_r.append(np.sum(recall)/len(recall))
        name = label_encoder.inverse_transform([class_])[0]
        results_AP[name] = [np.sum(precision)/len(precision),np.sum(recall)/len(recall),F1_score]

        print('------- Detection Scores - Class {0} ------------'.format(name))
        print('  AP:',np.sum(precision)/len(precision))
        print('  AR:',np.sum(recall)/len(recall))
        print('  F1 score:',F1_score)

    print(" ")
    print("----------- FINAL --------------")
    print("mAP:",np.mean(all_p))
    print("mAR:",np.mean(all_r))
    results_AP['mAP'] = np.mean(all_p)
    results_AP['mAR'] = np.mean(all_r)
