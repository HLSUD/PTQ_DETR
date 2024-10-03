import torch
import torch.nn as nn
import torchvision
from PIL import Image
import requests
import transformers
from transformers import AutoImageProcessor, DetrForObjectDetection, DetrForSegmentation
from transformers.image_transforms import rgb_to_id
import os, sys, io 
import numpy

def create_detr_based_model(model_name, task='detection'):
    if 'detr' in model_name.lower():
        if task == 'detection':
            image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
            model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        elif task == 'segmentation':
            image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50-panoptic")
            model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")
    return image_processor, model




if __name__ == '__main__':
    print('Getting Layers info...')
    task = 'detection'
    image_processor, model = create_detr_based_model('detr',task)
    print(model.__class__.__name__)
    blocks=[(name,m) for name,m in model.named_modules()]
    num_layer_dict = {'conv':0,'linear':0,'head':0}
    block_dict = []
    for name,m in blocks:
        if isinstance(m,nn.Conv2d):
            num_layer_dict['conv'] = num_layer_dict['conv'] + 1
        if isinstance(m,nn.Linear):
            num_layer_dict['linear'] = num_layer_dict['linear'] + 1
        sub_name = name.split('.')[-1]
        if sub_name not in block_dict:
            block_dict.append(sub_name)
    print(model)
    print(f"num of conv comp is {num_layer_dict['conv']}")
    print(f"num of linear comp is {num_layer_dict['linear']}")
    
