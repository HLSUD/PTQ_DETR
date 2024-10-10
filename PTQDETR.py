import torch
import torch.nn as nn
import torchvision
import transformers
from transformers import AutoImageProcessor, DetrForObjectDetection, DetrForSegmentation
from quant.q_conv import get_conv_quant_modules

class PTQDETR():
    def __init__(self, model_name, w_bit, a_bit):
        _, self.model = self.create_detr_based_model(model_name)
        self.w_bit = w_bit
        self.a_bit = a_bit

    def create_detr_based_model(self, model_name, task='detection'):
        """
            Get detr based models.
            Model List: [facebook/detr-resnet-50, facebook/detr-resnet-50-panoptic]
        """
        if 'detr' in model_name.lower():
            if task == 'detection':
                image_processor = AutoImageProcessor.from_pretrained(model_name)
                model = DetrForObjectDetection.from_pretrained(model_name)
            elif task == 'segmentation':
                image_processor = AutoImageProcessor.from_pretrained(model_name)
                model = DetrForSegmentation.from_pretrained(model_name)
        else:
            raise RuntimeError(f"Model {model_name} not supported.")
        print(f"Model {model_name} Loaded.")
        return image_processor, model
    
    def get_quantized_modules(self):
        wrapped_modules={}
        module_dict={}
        module_types = {"qkv":"qlinear_qkv", 
        "proj":'qlinear_proj', 
        'fc1':'qlinear_MLP_1', 
        'fc2':"qlinear_MLP_2", 
        'fc':'qlinear_MLP_1', 
        'head':'qlinear_classifier',
        'matmul1':"qmatmul_qk", 
        'matmul2':"qmatmul_scorev", 
        "reduction": "qlinear_reduction"}
        
        for name,m in self.model.named_modules():
            module_dict[name]=m
            idx=name.rfind('.')
            if idx==-1:
                idx=0
            father_name=name[:idx]
            if father_name in module_dict:
                print(name,father_name)
                father_module=module_dict[father_name]
            else:
                raise RuntimeError(f"father module {father_name} not found")
            if isinstance(m,nn.Conv2d):
                # Embedding Layer
                idx = idx+1 if idx != 0 else idx
                conv_module = get_conv_quant_modules(config, m)
                # new_m=get_conv_quant_modules("qconv",m.in_channels,m.out_channels,m.kernel_size,m.stride,m.padding,m.dilation,m.groups,m.bias is not None,m.padding_mode)
                conv_module.weight.data=m.weight.data
                conv_module.bias=m.bias
                replace_m=conv_module
                wrapped_modules[name] = conv_module
                setattr(father_module,name[idx:],replace_m)
            # elif isinstance(m,nn.Linear):
            #     # Linear Layer
            #     idx = idx+1 if idx != 0 else idx
            #     print(name,m)
            #     new_m = cfg.get_module(module_types[name[idx:]],m.in_features,m.out_features)
            #     new_m.weight.data=m.weight.data
            #     new_m.bias=m.bias
            #     replace_m=new_m
            #     wrapped_modules[name] = new_m
            #     setattr(father_module,name[idx:],replace_m)
            # elif isinstance(m,MatMul):
            #     # Matmul Layer
            #     idx = idx+1 if idx != 0 else idx
            #     new_m = cfg.get_module(module_types[name[idx:]])
            #     replace_m=new_m
            #     wrapped_modules[name] = new_m
            #     setattr(father_module,name[idx:],replace_m)
        print("Completed net wrap.")
        return wrapped_modules
    
    def model_calibration(self):
        return

if __name__ == '__main__':
    bit = 8
    q_detr = PTQDETR("facebook/detr-resnet-50", bit)
    q_detr.get_quantized_modules()