import yaml
import sys
import argparse
import os
from IPython import embed
from easydict import EasyDict
from interfaces.super_resolution import TextSR 
import matplotlib.pyplot as plt
import numpy as np 
from TP_Module import TP_module
import torch
from torch.autograd import Variable
from utils.metrics import get_string_crnn, get_string_aster
import warnings
warnings.filterwarnings('ignore')


def main(config, args):
    Mission = TextSR(config, args)
    Mission.train()
    
    # Mission = TextBase(config, args)
    # _, dataloader = Mission.get_train_data()
    # data = next(iter(dataloader))

    # images_hr, images_lr, interpolated_image_lr, label_strs = data

    # crnn = Mission.CRNN_init().eval()
    # parsed_images = Mission.parse_crnn_data((images_hr + 1) /2)
    # output = crnn(parsed_images)
    # preds = get_string_crnn(output)
    # print(preds)
    
    # Mission = TextBase(config, args)
    # _, dataloader = Mission.get_train_data()
    # data = next(iter(dataloader))

    # images_hr, images_lr, interpolated_image_lr, label_strs = data
    # moran = Mission.MORAN_init()
    # tensor, length, text, text_rev = Mission.parse_moran_data(images_hr)
    
    # print(tensor.shape, length.shape, text.shape, text_rev.shape)
    # output = moran(tensor, length, text, text_rev, test = True , debug = True)
    # preds, preds_reverse = output[0]
    # _, preds = preds.max(1)
    # sim_preds = Mission.converter_moran.decode(preds.data, length.data)
    # sim_preds = list(map(lambda x: x.strip().split('$')[0], sim_preds))
    # print(sim_preds)
    
    # Mission = TextBase(config, args)
    # _, dataloader = Mission.get_train_data()
    # data = next(iter(dataloader))

    # images_hr, images_lr, interpolated_image_lr, label_strs = data
    # aster, aster_info = Mission.Aster_init()
    # aster.eval()
    # input_dic  = Mission.parse_aster_data(images_hr)
    # return_dic = aster(input_dic)
    # pred_list, targ_list = get_string_aster(return_dic['output']['pred_rec'],input_dic['rec_targets'],aster_info)
    # print(pred_list)
    pass

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--arch', default='tp_maxVit', choices=['tsrn', 'bicubic', 'srcnn', 'vdsr', 'srres', 'esrgan', 'rdn',
                                                           'edsr', 'lapsrn'])
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--test_data_dir', type=str, default='../dataset/lmdb/str/TextZoom/test/medium/', help='')
    parser.add_argument('--batch_size', type=int, default=None, help='')
    parser.add_argument('--resume', type=str, default=None, help='')
    parser.add_argument('--vis_dir', type=str, default=None, help='')
    parser.add_argument('--rec', default='aster', choices=['aster', 'moran', 'crnn'])
    parser.add_argument('--STN', action='store_true', default=False, help='')
    parser.add_argument('--syn', action='store_true', default=False, help='use synthetic LR')
    parser.add_argument('--mixed', action='store_true', default=False, help='mix synthetic with real LR')
    parser.add_argument('--mask', action='store_true', default=False, help='')
    parser.add_argument('--gradient', action='store_true', default=False, help='')
    parser.add_argument('--hd_u', type=int, default=32, help='')
    parser.add_argument('--srb', type=int, default=5, help='')
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--demo_dir', type=str, default='./demo')
    args = parser.parse_args()
    config_path = os.path.join('config', 'super_resolution.yaml')
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config = EasyDict(config)
    main(config, args)