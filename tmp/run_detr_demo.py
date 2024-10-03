import torch
import torchvision
from PIL import Image
import requests
import transformers
from transformers import AutoImageProcessor, DetrForObjectDetection, DetrForSegmentation
from transformers.image_transforms import rgb_to_id
import os, sys, io 
import numpy

os.environ['HF_HOME'] = "/finance_ML/FinAi_Mapping_Knowledge/personal_data/liuhonghao/huggingface_cache"

def create_detr_based_model(model_name, task='detection'):
    if 'detr' in model_name.lower():
        if task == 'detection':
            image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
            model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        elif task == 'segmentation':
            image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50-panoptic")
            model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")
    return image_processor, model

def convert_model_output(outputs, size, task='detection'):
    if task == 'detection':
        # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=size)[0]
    elif task == 'segmentation':
        results = image_processor.post_process_panoptic_segmentation(outputs, target_sizes=[(300, 500)])
    return results

if __name__ == "__main__":
    print('DETR Testing...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load online image for testing
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # processing
    task = 'detection'
    image_processor, model = create_detr_based_model('detr',task)
    model = model.cuda()
    inputs = image_processor(images=image, return_tensors="pt")
    # inputs = inputs.to(device)
    # outputs = model(**inputs)

    pixel_values = inputs['pixel_values'].cuda()
    outputs = model(pixel_values)

    # print outputs
    size = torch.tensor([image.size[::-1]])
    results = convert_model_output(outputs, size, task)

    print("Results:")
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )

    # panoptic_seg = results[0]["segmentation"]
    # panoptic_segments_info = results[0]["segments_info"]
    # print(panoptic_segments_info)
    # print(panoptic_seg)
