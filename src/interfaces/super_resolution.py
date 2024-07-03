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
from utils.metrics import get_string_aster, get_string_crnn
import Levenshtein


def calculate_cer(str1, str2):
    return Levenshtein.distance(str1, str2) / len(str1)

def calculate_wer(str1, str2):
    words1 = str1.split()
    words2 = str2.split()
    return Levenshtein.distance(' '.join(words1), ' '.join(words2)) / len(words1)

def calculate_accuracy(str1, str2):
    if len(str1) == 0 or len(str2) == 0:
        return 0.0  # Handle case where str1 is empty
    correct_characters = sum(c1 == c2 for c1, c2 in zip(str1, str2))
    return correct_characters / len(str1)

class TextSR(TextBase):
    def __init__(self, config, args):
        super().__init__(config, args)
        self.model = model_sr().to(self.device)
        
        self.opt = self.optimizer_init(self.model)
        
        if self.config.TRAIN.resume:
            self.load_checkpoint(self.config.TRAIN.resume,self.config.TRAIN.new_lr)
            
        self.eval_models_dic = {'aster': self.Aster_init(),
                                'moran':self.MORAN_init(),
                                'crnn': self.CRNN_init()}
        
        train_dataset, train_loader = self.get_train_data()
        self.train_dataset = train_dataset
        self.train_loader = train_loader
        val_dataset, val_loader = self.get_val_data()
        self.val_dataset = val_dataset
        self.val_loader_easy = val_loader['easy']
        self.val_loader_medium = val_loader['medium']
        self.val_loader_hard = val_loader['hard']
        self.cri = TotalLoss(self.config)
        self.scheduler = LR_Scheduler(self.opt,self.config)
        self.train_convergence_list = []
        self.val_convergence_list = []
        self.epochs = self.config.TRAIN.epochs
        self.best_loss = float('inf')
        print(f'the training is done on {self.device}')
        
        
    def train(self):
        
        for epoch in range(1,self.epochs +1):
            epoch_losses = {
                'charbonnier_loss': 0,
                'kl_loss': 0,
                'l1_loss': 0,
                'total_loss': 0,
                'psnr' : 0,
                'ssim': 0,
            }
            
            for idx, (images_hr, images_lr, interpolated_image_lr, label_strs) in enumerate(tqdm(self.train_loader)):
                
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
                with torch.no_grad():
                    psnr = self.cal_psnr((sr_output+1) / 2,(images_hr + 1) / 2) # adding 1 and dividing by two to reverse the normalization
                    ssim = self.cal_ssim((sr_output+1) / 2,(images_hr + 1) / 2)
                    epoch_losses['psnr'] += psnr.item()
                    epoch_losses['ssim'] += ssim.item()


                #del images_hr, images_lr, interpolated_image_lr, sr_output, TP_lr, TP_hr, loss_dic
                #torch.cuda.empty_cache()
                
            num_batched = len(self.train_loader)
            epoch_losses['charbonnier_loss'] = epoch_losses['charbonnier_loss'] / num_batched
            epoch_losses['kl_loss'] = epoch_losses['kl_loss'] / num_batched
            epoch_losses['l1_loss'] = epoch_losses['l1_loss']/ num_batched
            epoch_losses['total_loss'] = epoch_losses['total_loss'] / num_batched
            epoch_losses['psnr'] /= num_batched
            epoch_losses['ssim'] /= num_batched
            self.train_convergence_list.append(epoch_losses)
            print(f'loss for epoch {epoch}')
            print('train loss: ')
            pprint(self.train_convergence_list[-1])
            self.eval_loss_metrics(epoch)
            print('validation loss: ')
            pprint(self.val_convergence_list[-1])
            print('--------------------------------')
            self.scheduler.step(epoch)
            if epoch % self.config.TRAIN.saveInterval == 0:
                if epoch_losses['total_loss'] < self.best_loss:
                    self.save_checkpoint(self.model,epoch,self.opt,self.epochs,epoch_losses,is_best=True)
                    self.best_loss = epoch_losses['total_loss']
                else:
                    self.save_checkpoint(self.model,epoch,self.opt,self.epochs,epoch_losses,is_best=False)
            if self.config.TRAIN.displayInterval % epoch:
                self.monitor_loss()
        
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
            for idx, (images_hr, images_lr, interpolated_image_lr, label_strs) in enumerate(self.val_loader_hard):
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
                val_losses['psnr'] += psnr.item()
                val_losses['ssim'] += ssim.item()
                
        
        num_batches = len(self.val_loader_hard)
        val_losses['charbonnier_loss'] /= num_batches
        val_losses['kl_loss'] /= num_batches
        val_losses['l1_loss'] /= num_batches
        val_losses['total_loss'] /= num_batches
        val_losses['psnr'] /= num_batches
        val_losses['ssim'] /= num_batches
                
        return val_losses
    
    def eval_OCR(self,):
        accuracies = {
            'crnn': 0,
            'moran': 0,
            'aster': 0,
        }
        
        with torch.no_grad():
            for idx, (images_hr, images_lr, interpolated_image_lr, label_strs) in enumerate(self.val_loader_easy):
                images_hr = images_hr.to(self.device)
                images_lr = images_lr.to(self.device)
                interpolated_image_lr = interpolated_image_lr.to(self.device)
                
                sr_output, _ = self.model(images_lr, interpolated_image_lr)
                _ = self.model.tp_module.generate_tp(images_hr)
                output_crnn = self.run_crnn(sr_output)
                output_moran = self.run_moran(sr_output)
                output_aster = self.run_aster(sr_output)
                
                acc_crnn  = 0
                acc_moran = 0
                acc_aster = 0
                
                for i in range(len(output_crnn)):
                    acc_crnn   += calculate_accuracy(label_strs[i], output_crnn[i])
                    acc_moran  += calculate_accuracy(label_strs[i], output_moran[i])
                    acc_aster  += calculate_accuracy(label_strs[i], output_aster[i])
                
                acc_crnn  /= len(output_crnn)
                acc_moran /= len(output_crnn)
                acc_aster /= len(output_crnn)
                
                accuracies['crnn']  += acc_crnn
                accuracies['moran'] += acc_moran
                accuracies['aster'] += acc_aster
                
            accuracies['crnn']  /= len(self.val_loader_easy)
            accuracies['moran'] /= len(self.val_loader_easy)
            accuracies['aster'] /= len(self.val_loader_easy)
            
            return accuracies


    def monitor_loss(self):
        
        os.makedirs(self.config.TRAIN.VAL.vis_dir, exist_ok=True)
        
        plot_path = os.path.join(self.config.TRAIN.VAL.vis_dir, 'loss_plot.png')


        epochs = range(1, len(self.train_convergence_list) + 1)
        
        plt.figure(figsize=(10, 6))
        total_loss_values_train = [d['total_loss'] for d in self.train_convergence_list]
        
        plt.plot(epochs, total_loss_values_train, label='Train Total Loss')
        total_loss_values_val = [d['total_loss'] for d in self.val_convergence_list]
        plt.plot(epochs, total_loss_values_val, label='Val Total Loss')
        
        plt.title('Training and Validation Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.grid(True)
        
        plt.savefig(plot_path)
        plt.close()
        
    def run_aster(self, images_sr):
        with torch.no_grad():
            aster, aster_info = self.eval_models_dic['aster']
            
            input_dic_sr = self.parse_aster_data(images_sr)
            
            return_dic_sr = aster(input_dic_sr)
            
            pred_list_sr, _ = get_string_aster(return_dic_sr['output']['pred_rec'], input_dic_sr['rec_targets'], aster_info)
            
            return pred_list_sr
        
    def run_crnn(self,image_sr):
        with torch.no_grad():
            parsed_images = self.parse_crnn_data((image_sr + 1) /2)
            output = self.eval_models_dic['crnn'](parsed_images)
            preds = get_string_crnn(output)
            return preds
    
    def run_moran(self,image_sr):
        with torch.no_grad():
            tensor, length, text, text_rev = self.parse_moran_data(image_sr)
            
            output = self.eval_models_dic['moran'](tensor, length, text, text_rev, test = True , debug = True)
            preds, _ = output[0]
            _, preds = preds.max(1)
            sim_preds = self.converter_moran.decode(preds.data, length.data)
            sim_preds = list(map(lambda x: x.strip().split('$')[0], sim_preds))
            return sim_preds