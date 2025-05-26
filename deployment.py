from cluster_test import ModelSelecter
from utils.modeltools import load_saved_cluster_models, load_single_model
from utils.util import setup_seed, non_max_suppression, scale, box_iou, compute_ap
import warnings
import torch
from tqdm import tqdm
import numpy
import pandas as pd
import argparse
import os
import yaml
from utils.dataloader import prepare_loader
import torchvision
from sklearn.decomposition import PCA
from nets.autoencoder import ConvAutoencoder
from sklearn.cluster import KMeans
import time
from contrastive_learner.contrastive_learner import ContrastiveLearner
import random
import numpy as np
import cv2
import copy
from utils import util


warnings.filterwarnings("ignore")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--args_file', default='utils/testing_args.yaml', type=str)
    parser.add_argument('--world_size', default=1, type=int)

    args = parser.parse_args()

    # args for DDP
    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    print(f"Local rank: {args.local_rank}")
    print(f"World size: {args.world_size}")

    # Setting random seed for reproducability
    # Seed is 0
    setup_seed()

    #Loading config
    with open(args.args_file) as cf_file:
        params = yaml.safe_load( cf_file.read())
    
    params['world_size'] = args.world_size
    params['local_rank'] = args.local_rank
    
    test_deployment(params)


def load_images(paths):
    images = []
    for path in paths:
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        images.append(image)
    tensor = convert_images_to_tensor(images)
    return images, tensor

def convert_images_to_tensor(images):
    transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
    tensors = []
    for image in images:
        tensor = transform(image)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.float()
        tensors.append(tensor)
    tensors = torch.cat(tensors, 0)
    return tensors

def extract_bounding_boxes(output, confidence_threshold):
    output = output.permute(1, 0)
        
    boxes_xywh = output[:, 0:4]
    confidences = output[:,4]
    class_scores = output[:,5:]
    
    class_conf, class_id = class_scores.max(1)
    final_confidence = confidences * class_conf
    
    keep = final_confidence > confidence_threshold
    boxes_xywh = boxes_xywh[keep]
    final_confidence = final_confidence[keep]
    class_id = class_id[keep]
    
    boxes_xyxy = torch.zeros_like(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2
    
    return boxes_xyxy, final_confidence, class_id

def non_max_suppresion(boxes_xyxy, final_confidence, class_id, iou_threshold):
    nms_indices = torchvision.ops.nms(boxes_xyxy, final_confidence, iou_threshold)
    boxes_xyxy = boxes_xyxy[nms_indices]
    final_confidence = final_confidence[nms_indices]
    class_id = class_id[nms_indices]
    
    return boxes_xyxy, final_confidence, class_id

def display_boxes(outputs, images, confidence_threshold, iou_threshold):
    working_images = copy.deepcopy(images)
    for outputid, output in enumerate(outputs):
        boxes_xyxy, final_confidence, class_id = extract_bounding_boxes(output, confidence_threshold)
        
        if boxes_xyxy.size(0) > 0:
            boxes_xyxy, final_confidence, class_id = non_max_suppresion(boxes_xyxy, final_confidence, class_id, iou_threshold)
        
            boxes_xyxy = boxes_xyxy.int()
            for i, box in enumerate(boxes_xyxy):
                x1, y1, x2, y2 = box
                cv2.rectangle(working_images[outputid], (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(working_images[outputid], f'Class {class_id[i]}', (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    return working_images

def test_deployment(args):
    selector = ModelSelecter(args)
    
    paths = [
        "Data/images/test/20210111_clip_50_2310_image_0017.jpg",
        "Data/images/test/20200530_clip_21_1748_image_0017.jpg",
        "Data/images/test/20200825_clip_2_1312_image_0035.jpg",
        "Data/images/test/20200625_clip_5_0220_image_0063.jpg",
        "Data/images/test/20210212_clip_8_2306_image_0052.jpg",
        "Data/images/test/20200813_clip_32_1407_image_0017.jpg",
        "Data/images/test/20200822_clip_16_0707_image_0115.jpg",
        "Data/images/test/20210112_clip_30_1719_image_0099.jpg",
        "Data/images/test/20210115_clip_43_2251_image_0002.jpg",
        "Data/images/test/20200709_clip_33_1427_image_0075.jpg"
            ]
    
    
    iou_threshold = 0.5
    confidence_threshold = 0.0001
    
    images, tensor = load_images(paths)
    tensor = tensor.cuda()
    tensor = tensor.half()

    outputs = selector.make_predictions(tensor)
    box_images = display_boxes(outputs, images, confidence_threshold, iou_threshold)
    
    for i, image in enumerate(box_images):
        cv2.imshow(f"{i}", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()