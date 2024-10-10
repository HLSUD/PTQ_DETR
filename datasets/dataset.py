# DETR dataset

import os
import json
import numpy as np
import copy
import warnings
import random
import torch
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor
import datasets.transforms as T
from pycocotools.coco import COCO
from PIL import Image
import PIL
from PIL import ImageFile
import torchvision


# ---- Contents below copy from Facebook DETR https://github.com/facebookresearch/detr/blob/main/datasets/coco.py
def make_coco_transforms(image_set, fix_size=False, strong_aug=False, args=None):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # config the params for data aug
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    max_size = 1333
    scales2_resize = [400, 500, 600]
    scales2_crop = [384, 600]
    
    # update args from config files
    scales = getattr(args, 'data_aug_scales', scales)
    max_size = getattr(args, 'data_aug_max_size', max_size)
    scales2_resize = getattr(args, 'data_aug_scales2_resize', scales2_resize)
    scales2_crop = getattr(args, 'data_aug_scales2_crop', scales2_crop)

    # resize them
    data_aug_scale_overlap = getattr(args, 'data_aug_scale_overlap', None)
    if data_aug_scale_overlap is not None and data_aug_scale_overlap > 0:
        data_aug_scale_overlap = float(data_aug_scale_overlap)
        scales = [int(i*data_aug_scale_overlap) for i in scales]
        max_size = int(max_size*data_aug_scale_overlap)
        scales2_resize = [int(i*data_aug_scale_overlap) for i in scales2_resize]
        scales2_crop = [int(i*data_aug_scale_overlap) for i in scales2_crop]

    datadict_for_print = {
        'scales': scales,
        'max_size': max_size,
        'scales2_resize': scales2_resize,
        'scales2_crop': scales2_crop
    }
    print("data_aug_params:", json.dumps(datadict_for_print, indent=2))
        

    if image_set == 'train':
        if fix_size:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomResize([(max_size, max(scales))]),
                # T.RandomResize([(512, 512)]),
                normalize,
            ])

        if strong_aug:
            import datasets.sltransform as SLT
            
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose([
                        T.RandomResize(scales2_resize),
                        T.RandomSizeCrop(*scales2_crop),
                        T.RandomResize(scales, max_size=max_size),
                    ])
                ),
                SLT.RandomSelectMulti([
                    SLT.RandomCrop(),
                    SLT.LightingNoise(),
                    SLT.AdjustBrightness(2),
                    SLT.AdjustContrast(2),
                ]),
                normalize,
            ])
        
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize(scales2_resize),
                    T.RandomSizeCrop(*scales2_crop),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
            normalize,
        ])

    if image_set in ['val', 'eval_debug', 'train_reg', 'test']:

        if os.environ.get("GFLOPS_DEBUG_SHILONG", False) == 'INFO':
            print("Under debug mode for flops calculation only!!!!!!!!!!!!!!!!")
            return T.Compose([
                T.ResizeDebug((1280, 800)),
                normalize,
            ])   

        return T.Compose([
            T.RandomResize([max(scales)], max_size=max_size),
            normalize,
        ])



    raise ValueError(f'unknown {image_set}')

class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, aux_target_hacks=None):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.aux_target_hacks = aux_target_hacks

    def change_hack_attr(self, hackclassname, attrkv_dict):
        target_class = dataset_hook_register[hackclassname]
        for item in self.aux_target_hacks:
            if isinstance(item, target_class):
                for k,v in attrkv_dict.items():
                    setattr(item, k, v)

    def get_hack(self, hackclassname):
        target_class = dataset_hook_register[hackclassname]
        for item in self.aux_target_hacks:
            if isinstance(item, target_class):
                return item

    def __getitem__(self, idx):
        """
        Output:
            - target: dict of multiple items
                - boxes: Tensor[num_box, 4]. \
                    Init type: x0,y0,x1,y1. unnormalized data.
                    Final type: cx,cy,w,h. normalized data. 
        """
        try:
            img, target = super(CocoDetection, self).__getitem__(idx)
        except:
            print("Error idx: {}".format(idx))
            idx += 1
            img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        # convert to needed format
        if self.aux_target_hacks is not None:
            for hack_runner in self.aux_target_hacks:
                target, img = hack_runner(target, img=img)

        return img, target

# -------------- Coco dataloader ----------


class COCOLoaderGenerator():
    def __init__(self, data_dir, ann_file,  train_batch_size=1, test_batch_size=1, num_workers=0, kwargs={}):
        self.data_dir = data_dir
        self.ann_file = ann_file
        self.train_batch_size=train_batch_size
        self.test_batch_size=test_batch_size
        self.num_workers=num_workers
        self.kwargs=kwargs
        self.train_loader_kwargs = {
        'num_workers': self.num_workers ,
        'pin_memory': kwargs.get('pin_memory',True),
        'drop_last':kwargs.get('drop_last',False)
        }
        self.test_loader_kwargs=self.train_loader_kwargs.copy()
        self.load()

    def load(self):
        # download from https://github.com/pjreddie/darknet/tree/master/scripts/get_coco_dataset.sh
        # self.train_set = CocoDetection(data_dir, ann_file=annotation_file, transforms=make_coco_transforms('train'), return_masks=False)
        self.test_set = CocoDetection(data_dir, ann_file=annotation_file, transforms=make_coco_transforms('val'), return_masks=False)
    
    def get_train_loader(self):
        train_dataloader =  DataLoader(self.train_set,  batch_size=self.train_batch_size, shuffle=True,  **self.train_loader_kwargs)
        return train_dataloader
    
    def get_val_loader(self):
        val_dataloader =  DataLoader(self.test_set,  batch_size=self.test_batch_size, shuffle=False,  **self.test_loader_kwargs)
        return val_dataloader

    def calib_loader(self,num=1024,seed=3):
        if self._calib_set is None:
            np.random.seed(seed)
            inds=np.random.permutation(len(self.train_set))[:num]
            self._calib_set=torch.utils.data.Subset(copy.deepcopy(self.train_set),inds)
            self._calib_set.dataset.transform=self.test_transform
        return DataLoader(self._calib_set, batch_size=num, shuffle=False,  **self.train_loader_kwargs)

if __name__ == "__main__":
    # Example usage:
    data_dir = '/finance_ML/FinAi_Mapping_Knowledge/personal_data/liuhonghao/datasets/COCO2017/val2017'   # Path to the images
    annotation_file = '/finance_ML/FinAi_Mapping_Knowledge/personal_data/liuhonghao/datasets/COCO2017/annotations/instances_val2017.json'  # Path to the annotations


    # coco_dataset = CocoDetection(data_dir, ann_file=annotation_file, transforms=make_coco_transforms('val'), return_masks=False)

    # Create the DataLoader
    coco_loader = COCOLoaderGenerator(data_dir, annotation_file, test_batch_size=1, num_workers=1)
    coco_dataloader = coco_loader.get_val_loader()
    # Iterating through the dataset
    for images, targets in coco_dataloader:
        print(images.shape)  # Shape of the batch of images
        print(targets)        # List of annotations for each image in the batch
        break
