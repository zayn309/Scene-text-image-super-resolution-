import yaml
import sys
import argparse
import os
from IPython import embed
from easydict import EasyDict
from interfaces.base import TextBase 
import matplotlib.pyplot as plt
import numpy as np 
from TP_Module import TP_module


def main(config, args):
    Mission = TextBase(config, args)
    _, dataloader = Mission.get_train_data()
    data = next(iter(dataloader))

    images_hr, images_lr,interpolated_image_lr, label_strs = data
    
    tp_generator = TP_module()
    probs, tp_features = tp_generator(interpolated_image_lr)
    print(probs.shape)
    print(tp_features.shape)
    
    # # Convert tensors to numpy arrays
    # image_hr = images_hr[0].numpy()
    # image_lr = interpolated_image_lr[0].numpy()
    
    # # If images have 3 channels (RGB), transpose them
    # if image_hr.shape[0] == 3:
    #     image_hr = np.transpose(image_hr, (1, 2, 0))
    # if image_lr.shape[0] == 3:
    #     image_lr = np.transpose(image_lr, (1, 2, 0))
    
    # # Plot HR and LR images side by side
    # fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    # axes[0, 0].imshow(image_hr)
    # axes[0, 0].set_title('High Resolution')
    # axes[0, 0].axis('off')

    # axes[0, 1].imshow(image_lr)
    # axes[0, 1].set_title('Low Resolution')
    # axes[0, 1].axis('off')
    
    # # Flatten the images to get the pixel values
    # image_hr_flat = image_hr.flatten()
    # image_lr_flat = image_lr.flatten()
    
    # # Plot histograms of the pixel values
    # axes[1, 0].hist(image_hr_flat, bins=256, color='blue', alpha=0.7)
    # axes[1, 0].set_title('Pixel Value Distribution (HR)')
    # axes[1, 0].set_xlabel('Pixel Value')
    # axes[1, 0].set_ylabel('Frequency')
    
    # axes[1, 1].hist(image_lr_flat, bins=256, color='green', alpha=0.7)
    # axes[1, 1].set_title('Pixel Value Distribution (LR)')
    # axes[1, 1].set_xlabel('Pixel Value')
    # axes[1, 1].set_ylabel('Frequency')
    
    # fig.suptitle(f'Label: {label_strs[0]}')
    # plt.savefig('fig2.png')
    # plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--arch', default='tsrn', choices=['tsrn', 'bicubic', 'srcnn', 'vdsr', 'srres', 'esrgan', 'rdn',
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