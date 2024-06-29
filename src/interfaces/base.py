import torch
import sys
import os
from tqdm import tqdm
import math
import torch.nn as nn
import torch.optim as optim
from IPython import embed
import math
import cv2
import string
from PIL import Image
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from collections import OrderedDict
import matplotlib.pyplot as plt 

from model import recognizer
from model import moran
from model import crnn

from dataset import lmdbDataset, alignCollate_real, ConcatDataset, lmdbDataset_real, alignCollate_syn, lmdbDataset_mix

from utils.labelmaps import get_vocabulary, labels2strs
from Model import model_sr

sys.path.append('../')
from utils import util, ssim_psnr, utils_moran, utils_crnn
import dataset.dataset as dataset

class TextBase(object):
    def __init__(self, config, args):
        super(TextBase, self).__init__()
        self.config = config
        self.args = args
        self.scale_factor = self.config.TRAIN.down_sample_scale
        if self.args.syn:
            self.align_collate = alignCollate_syn
            self.load_dataset = lmdbDataset
        elif self.args.mixed:
            self.align_collate = alignCollate_real
            self.load_dataset = lmdbDataset_mix
        else:
            self.align_collate = alignCollate_real
            self.load_dataset = lmdbDataset_real
        self.resume = args.resume if args.resume is not None else config.TRAIN.resume
        self.batch_size = args.batch_size if args.batch_size is not None else self.config.TRAIN.batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        alpha_dict = {
            'digit': string.digits,
            'lower': string.digits + string.ascii_lowercase,
            'upper': string.digits + string.ascii_letters,
            'all': string.digits + string.ascii_letters + string.punctuation
        }
        self.test_data_dir = self.args.test_data_dir if self.args.test_data_dir is not None else self.config.TEST.test_data_dir
        self.voc_type = self.config.TRAIN.voc_type
        self.alphabet = alpha_dict[self.voc_type]
        self.max_len = config.TRAIN.max_len
        self.vis_dir = self.args.vis_dir if self.args.vis_dir is not None else self.config.TRAIN.VAL.vis_dir
        self.cal_psnr = ssim_psnr.calculate_psnr
        self.cal_ssim = ssim_psnr.SSIM()
        self.mask = self.args.mask
        alphabet_moran = ':'.join(string.digits+string.ascii_lowercase+'$')
        self.converter_moran = utils_moran.strLabelConverterForAttention(alphabet_moran, ':')
        self.converter_crnn = utils_crnn.strLabelConverter(string.digits + string.ascii_lowercase)
        
    def get_train_data(self):
        cfg = self.config.TRAIN
        if isinstance(cfg.train_data_dir, list):
            dataset_list = []
            for data_dir_ in cfg.train_data_dir:
                dataset_list.append(
                    self.load_dataset(root=data_dir_,
                                      voc_type=cfg.voc_type,
                                      max_len=cfg.max_len))
            train_dataset = ConcatDataset(dataset_list)
        else:
            raise TypeError('check trainRoot')

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=int(cfg.workers),
            collate_fn=self.align_collate(imgH=cfg.height, imgW=cfg.width, down_sample_scale=cfg.down_sample_scale,
                                          mask=self.mask),
            drop_last=True)
        return train_dataset, train_loader
    #working 
    def get_val_data(self):
        cfg = self.config.TRAIN
        assert isinstance(cfg.VAL.val_data_dir, list)
        dataset_dic = {}
        loader_dic = {}
        for data_dir_ in cfg.VAL.val_data_dir:
            name = os.path.basename(data_dir_)
            val_dataset, val_loader = self.get_test_data(data_dir_)
            dataset_dic[name] = val_dataset
            loader_dic[name] = val_loader
        return dataset_dic, loader_dic
    # working   
    def get_test_data(self, dir_):
        cfg = self.config.TRAIN
        self.args.test_data_dir
        test_dataset = self.load_dataset(root=dir_,
                                         voc_type=cfg.voc_type,
                                         max_len=cfg.max_len,
                                         test=True,
                                         )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=int(cfg.workers),
            collate_fn=self.align_collate(imgH=cfg.height, imgW=cfg.width, down_sample_scale=cfg.down_sample_scale,
                                          mask=self.mask),
            drop_last=False)
        return test_dataset, test_loader

    def generator_init(self):
        cfg = self.config.TRAIN
        model = model_sr()
        return model
        
    def optimizer_init(self, model):
        cfg = self.config.TRAIN
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr,
                               betas=(cfg.beta1, 0.999))
        return optimizer

    def visualize_and_save(self,LR, HR, SR, pred_str_lr, pred_str_sr, label_strs_hr,epoch, base_dir='visualizations'):
        """
        Visualize the first 2 LR, HR, and SR images with their corresponding labels and save the figure as a PNG file.
        
        Parameters:
        - LR: Batch tensor of Low-Resolution images.
        - HR: Batch tensor of High-Resolution images.
        - SR: Batch tensor of Super-Resolution images.
        - pred_str_lr: List of predicted strings for LR images.
        - pred_str_sr: List of predicted strings for SR images.
        - label_strs_hr: List of true label strings for HR images.
        - epoch: The current epoch number.
        - base_dir: Base directory to save the figures (default is 'visualizations').
        """
        
        # Ensure the directory for the current epoch exists
        epoch_dir = os.path.join(base_dir, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)
        
        # Only take the first 2 images
        num_images = min(2, LR.shape[0])
        
        # Move tensors to CPU if they are on CUDA
        if LR.is_cuda:
            LR = LR.cpu().permute(1,2,0)
        if HR.is_cuda:
            HR = HR.cpu().permute(1,2,0)
        if SR.is_cuda:
            SR = SR.cpu().permute(1,2,0)
        
        fig, axes = plt.subplots(3, num_images, figsize=(10, 10))
        
        for i in range(num_images):
            # LR images and their labels
            axes[0, i].imshow(LR[i].numpy(), cmap='gray')
            axes[0, i].set_title(f"LR: {pred_str_lr[i]}")
            axes[0, i].axis('off')
            
            # HR images and their labels
            axes[1, i].imshow(HR[i].numpy(), cmap='gray')
            axes[1, i].set_title(f"HR: {label_strs_hr[i]}")
            axes[1, i].axis('off')
            
            # SR images and their labels
            axes[2, i].imshow(SR[i].numpy(), cmap='gray')
            axes[2, i].set_title(f"SR: {pred_str_sr[i]}")
            axes[2, i].axis('off')
        
        plt.tight_layout()
        filename = os.path.join(epoch_dir, 'visualization.png')
        plt.savefig(filename)
        plt.show()
    
    def save_checkpoint(self, net, epoch,opt, iters, best_acc_dict, is_best):
        ckpt_path = os.path.join('ckpt', self.vis_dir)
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)
        save_dict = {
            'state_dict': net.module.state_dict(),
            'optimizer' : opt.state_dict(),
            'info': {'arch': self.args.arch, 'iters': iters, 'epochs': epoch, 'batch_size': self.batch_size,
                     'voc_type': self.voc_type, 'up_scale_factor': self.scale_factor},
            'best_history_res': best_acc_dict,
            'param_num': sum([param.nelement() for param in net.module.parameters()]),
        }
        if is_best:
            torch.save(save_dict, os.path.join(ckpt_path, 'model_best.pth'))
        else:
            torch.save(save_dict, os.path.join(ckpt_path, 'checkpoint.pth'))
    
        
    def load_checkpoint(self, checkpoint_file, lr = None):
        print("=> Loading checkpoint")
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["state_dict"])
        self.opt.load_state_dict(checkpoint["optimizer"])

        # If we don't do this then it will just have learning rate of old checkpoint
        # and it will lead to many hours of debugging \:
        if lr :
            for param_group in self.opt.param_groups:
                param_group["lr"] =  lr

    def MORAN_init(self):
        cfg = self.config.TRAIN
        alphabet = ':'.join(string.digits + string.ascii_lowercase + '$')
        MORAN = moran.MORAN(1, len(alphabet.split(':')), 256, 32, 100, BidirDecoder=True, inputDataType='torch.FloatTensor', CUDA=False)
        
        model_path = self.config.TRAIN.VAL.moran_pretrained
        print('loading pre-trained moran model from %s' % model_path)
        
        # Load the model to CPU
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        MORAN_state_dict_rename = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")  # remove `module.`
            MORAN_state_dict_rename[name] = v
        
        MORAN.load_state_dict(MORAN_state_dict_rename)
        MORAN = MORAN.to(self.device)  # Ensure the device is set correctly
        
        # DataParallel is not necessary for CPU, but if you want to keep it, ensure it's set for CPU
        if torch.cuda.device_count() > 1:
            MORAN = torch.nn.DataParallel(MORAN)
        
        for p in MORAN.parameters():
            p.requires_grad = False
        
        MORAN.eval()
        return MORAN.eval()

    def parse_moran_data(self, imgs_input):
        batch_size = imgs_input.shape[0]
        imgs_input = torch.nn.functional.interpolate(imgs_input, (32, 100), mode='bicubic')
        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        text = torch.LongTensor(batch_size * 5)
        length = torch.IntTensor(batch_size)
        max_iter = 20
        t, l = self.converter_moran.encode(['0' * max_iter] * batch_size)
        utils_moran.loadData(text, t)
        utils_moran.loadData(length, l)
        return tensor, length, text, text

    def CRNN_init(self):
        model = crnn.CRNN(32, 1, 37, 256)
        model = model.to(self.device)
        model_path = self.config.TRAIN.VAL.crnn_pretrained
        print('loading pretrained crnn model from %s' % model_path)
        model.load_state_dict(torch.load(model_path))
        return model.eval()

    def parse_crnn_data(self, imgs_input):
        imgs_input = torch.nn.functional.interpolate(imgs_input, (32, 100), mode='bicubic')
        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        return tensor

    def Aster_init(self):
        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        aster = recognizer.RecognizerBuilder(
            arch='ResNet_ASTER',
            rec_num_classes=aster_info.rec_num_classes,
            sDim=512,
            attDim=512,
            max_len_labels=aster_info.max_len,
            eos=aster_info.char2id[aster_info.EOS],
            STN_ON=True
        )

        # Determine the device to load the model on
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the state dict on the appropriate device
        state_dict = torch.load(self.config.TRAIN.VAL.aster_pretrained, map_location=device)['state_dict']
        aster.load_state_dict(state_dict)
        
        print('Loaded pre-trained aster model from %s' % self.config.TRAIN.VAL.aster_pretrained)
        
        # Move the model to the appropriate device
        aster = aster.to(device)

        
        return (aster.eval(), aster_info)

    def parse_aster_data(self, imgs_input):
        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        input_dict = {}
        images_input = imgs_input.to(self.device)
        input_dict['images'] = images_input
        batch_size = images_input.shape[0]
        input_dict['rec_targets'] = torch.IntTensor(batch_size, aster_info.max_len).fill_(1)
        input_dict['rec_lengths'] = [aster_info.max_len] * batch_size
        return input_dict

class AsterInfo(object):
    def __init__(self, voc_type):
        super(AsterInfo, self).__init__()
        self.voc_type = voc_type
        assert voc_type in ['digit', 'lower', 'upper', 'all']
        self.EOS = 'EOS'
        self.max_len = 100
        self.PADDING = 'PADDING'
        self.UNKNOWN = 'UNKNOWN'
        self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
        self.char2id = dict(zip(self.voc, range(len(self.voc))))
        self.id2char = dict(zip(range(len(self.voc)), self.voc))
        self.rec_num_classes = len(self.voc)

    def __str__(self):
        return (f"AsterInfo(voc_type={self.voc_type}, EOS={self.EOS}, max_len={self.max_len}, "
                f"PADDING={self.PADDING}, UNKNOWN={self.UNKNOWN}, voc_size={len(self.voc)}, "
                f"rec_num_classes={self.rec_num_classes})")
