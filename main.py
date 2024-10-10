from datasets.dataset import COCOLoaderGenerator
from PTQDETR import PTQDETR
import argparse
import configs.config as config

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str)
    # parser.add_argument('--options',
    #     nargs='+',
    #     action=DictAction,
    #     help='override some settings in the used config, the key-value pair '
    #     'in xxx=yyy format will be merged into config file.') ## check

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, default='/datastes/COCO2017/')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--fix_size', action='store_true')

    parser.add_argument('--model_name', type=str, default='facebook/detr-resnet-50')


    # quant parameters
    parser.add_argument('--w_bit', type=int, default=8,
                        help='weight number of bit')
    parser.add_argument('--a_bit', type=int, default=8,
                        help='activation number of bit')
    parser.add_argument('--calib_size', type=int, default=32,
                        help='calibration size')
    
    # running config
    parser.add_argument('--device', default='cuda',
                        help='device to use for quantization')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--save_log', action='store_true')
    return parser

def main(args):
    # create quantized DETR based models
    ptqdetr = PTQDETR("facebook/detr-resnet-50", config.w_bit, config.a_bit)
    q_modules = ptqdetr.get_quantized_modules()
    # load detection datasets
    calib_size = config.calib_size
    coco_loader = COCOLoaderGenerator(config.dataset_dir, config.annotation_file, 
                                      test_batch_size=config.batch_size, num_workers=config.num_workers)
    coco_val_loader=coco_loader.get_val_loader()
    calib_loader=coco_loader.get_calib_loader(num=calib_size)
    # calibration

    # evaluation

if __name__ == "__main__":
    print("Begin the qunatization...")
    parser = argparse.ArgumentParser('DETR-Based Model Quantization', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

