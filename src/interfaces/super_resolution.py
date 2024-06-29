from interfaces.base import TextBase
import torch
import torch.nn as nn
import torch.nn.functional as F
from Model import model_sr
from Loss.general_loss import TotalLoss 
from LR_schedualer.LR_scheduler import LR_Scheduler
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import gc
from pprint import pprint
from utils.metrics import get_string_aster


class TextSR(TextBase):
    def __init__(self, config, args):
        super().__init__(config, args)
        self.model = model_sr().to(self.device)
        
        self.opt = self.optimizer_init(self.model)
        
        if self.config.TRAIN.resume:
            self.load_checkpoint(self.config.TRAIN.resume,self.config.TRAIN.new_lr)
            
        self.eval_models_dic = {'aster': self.Aster_init(),}
        #                         'moran':self.MORAN_init(),
        #                         'crnn': self.CRNN_init()}
        
        train_dataset, train_loader = self.get_train_data()
        self.train_dataset = train_dataset
        self.train_loader = train_loader
        val_dataset, val_loader = self.get_val_data()
        self.val_dataset = val_dataset
        self.val_loader = val_loader
        self.val_dataloader_medium = self.val_loader['medium']
        self.cri = TotalLoss(self.config)
        self.scheduler = LR_Scheduler(self.opt,self.config)
        self.train_convergence_list = []
        self.val_convergence_list = []
        self.epochs = self.config.TRAIN.epochs
        self.best_loss = float('inf')
        print(f'the training is done on {self.device}')
        
    def train(self):
        loop = tqdm(self.train_loader, leave=True)
        
        for epoch in range(1,self.epochs +1):
            epoch_losses = {
                'charbonnier_loss': 0,
                'kl_loss': 0,
                'l1_loss': 0,
                'total_loss': 0,
                'psnr' : 0,
                'ssim': 0,
            }
            for idx, (images_hr, images_lr, interpolated_image_lr, label_strs) in enumerate(loop):
                
                images_hr = images_hr.to(self.device)
                images_lr = images_lr.to(self.device)
                interpolated_image_lr = interpolated_image_lr.to(self.device)
                
                sr_output, TP_lr = self.model(images_lr, interpolated_image_lr)
                TP_hr = self.model.tp_module.generate_tp(images_hr)
                
                loss_dic = self.cri(sr_output,images_hr,TP_lr,TP_hr)
                
                loss = loss_dic['total_loss']
                
                self.opt.zero_grad()
                
                loss.backward()
                epoch_losses['charbonnier_loss'] += loss_dic['charbonnier_loss']
                epoch_losses['kl_loss'] += loss_dic['kl_loss']
                epoch_losses['l1_loss'] += loss_dic['l1_loss']
                epoch_losses['total_loss'] += loss.item()
                
                
                # Gradient descent or adam step
                self.opt.step()
                
                psnr = self.cal_psnr((sr_output+1) / 2,(images_hr + 1) / 2) # adding 1 and dividing by two to reverse the normalization
                ssim = self.cal_ssim((sr_output+1) / 2,(images_hr + 1) / 2)
                epoch_losses['psnr'] += psnr
                epoch_losses['ssim'] += ssim
                torch.cuda.empty_cache()
                
                
                
            num_batched = len(self.train_loader)
            epoch_losses['charbonnier_loss'] = epoch_losses['charbonnier_loss'] / num_batched
            epoch_losses['kl_loss'] = epoch_losses['kl_loss']/ num_batched
            epoch_losses['l1_loss'] = epoch_losses['l1_loss']/ num_batched
            epoch_losses['total_loss'] = epoch_losses['total_loss'] / num_batched
            epoch_losses['psnr'] /= num_batched
            epoch_losses['ssim'] /= num_batched
            
            self.train_convergence_list.append(epoch_losses)
            self.eval_loss()
            print(f'loss for epoch {epoch}')
            print('train loss: ')
            pprint(self.train_convergence_list[-1])
            print('validation loss: ')
            pprint(self.val_convergence_list[-1])
            print('--------------------------------')
            self.scheduler.step(epoch)
            if epoch % self.config.TRAIN.saveInterval == 0:
                if epoch_losses['total_loss'] < self.best_loss:
                    self.save_checkpoint(self.model,epoch,self.opt,self.epochs,epoch_losses,is_best=True)
                else:
                    self.save_checkpoint(self.model,epoch,self.opt,self.epochs,epoch_losses,is_best=False)
        
    def eval_loss_metrics(self,epoch):
        self.model.eval()  # Set the model to evaluation mode
        val_losses = {
            'charbonnier_loss': 0,
            'kl_loss': 0,
            'l1_loss': 0,
            'total_loss': 0,
            'psnr' : 0,
            'ssim': 0,
        }
        
        with torch.no_grad():
            for idx, (images_hr, images_lr, interpolated_image_lr, label_strs) in enumerate(self.val_dataloader_medium):
                images_hr = images_hr.to(self.device)
                images_lr = images_lr.to(self.device)
                interpolated_image_lr = interpolated_image_lr.to(self.device)
                
                sr_output, TP_lr = self.model(images_lr, interpolated_image_lr)
                TP_hr = self.model.tp_module.generate_tp(images_hr)
                
                if epoch % self.config.TRAIN.displayInterval == 0 and idx == 0:
                    returned_str = self.run_aster(images_hr[0:2],interpolated_image_lr[0:2],sr_output[0:2])
                    self.visualize_and_save(images_lr[0:2],images_hr[0:2],sr_output[0:2],returned_str['lr'],returned_str['sr'],returned_str['hr'],epoch)
                    
                loss_dic = self.cri(sr_output, images_hr, TP_lr, TP_hr)
                
                loss = loss_dic['total_loss']
                psnr = self.cal_psnr((sr_output+1) / 2,(images_hr + 1) / 2) # adding 1 and dividing by two to reverse the normalization
                ssim = self.cal_ssim((sr_output+1) / 2,(images_hr + 1) / 2)
                
                val_losses['charbonnier_loss'] += loss_dic['charbonnier_loss']
                val_losses['kl_loss'] += loss_dic['kl_loss']
                val_losses['l1_loss'] += loss_dic['l1_loss']
                val_losses['total_loss'] += loss.item()
                val_losses['psnr'] += psnr
                val_losses['ssim'] += ssim
                
        
        num_batches = len(self.val_dataloader_medium)
        val_losses['charbonnier_loss'] /= num_batches
        val_losses['kl_loss'] /= num_batches
        val_losses['l1_loss'] /= num_batches
        val_losses['total_loss'] /= num_batches
        val_losses['psnr'] /= num_batches
        val_losses['ssim'] /= num_batches
                
        
        self.val_convergence_list.append(val_losses)
        self.model.train()
        
    def eval_OCR(self,):
        pass

    def monitor_loss(self):
        
        plot_path = os.path.join(self.config.TRAIN.VAL.vis_dir, 'loss_plot.png')
        epochs = range(1, len(self.train_convergence_list) + 1)
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(epochs, self.train_convergence_list['total_loss'], label='Train Total Loss')
        
        plt.plot(epochs, self.val_convergence_list['total_loss'], label='Val Total Loss')
        
        plt.title('Training and Validation Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.grid(True)
        
        # Save the plot as PNG
        plt.savefig(plot_path)
        
    def run_aster(self, images_hr, images_lr, images_sr):
        OCR_output = {}
        with torch.no_grad():
            aster, aster_info = self.eval_OCR['aster']
            
            input_dic_lr = self.parse_aster_data(images_lr)
            input_dic_hr = self.parse_aster_data(images_hr)
            input_dic_sr = self.parse_aster_data(images_sr)
            
            return_dic_lr = aster(input_dic_lr)
            return_dic_hr = aster(input_dic_hr)
            return_dic_sr = aster(input_dic_sr)
            
            pred_list_lr, _ = get_string_aster(return_dic_lr['output']['pred_rec'], input_dic_lr['rec_targets'], aster_info)
            pred_list_hr, _ = get_string_aster(return_dic_hr['output']['pred_rec'], input_dic_hr['rec_targets'], aster_info)
            pred_list_sr, _ = get_string_aster(return_dic_sr['output']['pred_rec'], input_dic_sr['rec_targets'], aster_info)
            
            # print('LR predictions:', pred_list_lr)
            # print('HR predictions:', pred_list_hr)
            # print('SR predictions:', pred_list_sr)
            OCR_output['lr'] = pred_list_lr
            OCR_output['hr'] = pred_list_hr
            OCR_output['sr'] = pred_list_sr
            return OCR_output