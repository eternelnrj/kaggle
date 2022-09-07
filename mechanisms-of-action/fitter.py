import os
import subprocess
from path import Path
import time

from datetime import datetime
from glob import glob

import torch
import torch.nn.functional as F

from network import make_model
from dataloaders import get_trainloaders
from loss import LossMeter, AverageMeter

class Fitter:
    def __init__(self, config):
        
        self.current_dir = subprocess.check_output("pwd", shell=True).decode('ascii').strip()
        self.checkpoint_dir = os.path.join(self.current_dir, 'checkpoints')  
        
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            
        self.config = config
        self.epoch = 0
        self.n_epochs = config.n_epochs

        self.model = make_model(config).to(config.device)
        self.model_name = config.model_name
            
        self.device = config.device
             
        self.optimizer = config.optimizer_class(self.model.parameters(), **config.optimizer_params)
        self.criterion = config.criterion_class(**config.criterion_params)        

        self.log_dir = os.path.join(self.current_dir, 'logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        self.log_path = f'{self.log_dir}/logs.txt'
        
        
        self.log(f'Fitter prepared. Device is {self.device}')
        
        self.best_summary_score = -10**5
        self.best_summary_loss = 10 ** 5

    ###############################################################################################################    
    
    def reduce_lr(self, factor=2):
        for i in range(len(self.optimizer.param_groups)):
            self.optimizer.param_groups[i]['lr'] = self.optimizer.param_groups[i]['lr'] / factor
            
    def increase_lr(self, factor=2):
        for i in range(len(self.optimizer.param_groups)):
            self.optimizer.param_groups[i]['lr'] = self.optimizer.param_groups[i]['lr'] * factor
        
    #############################################################################################################
     
    def train_k_fold(self, features, targets,  total_folds, continue_from_checkpoint=False):
        for number_fold in range(total_folds):
            print(f"Fold number {number_fold}/{total_folds}")
            self.reset()
            self.update_dir(total_folds, number_fold)
            if continue_from_checkpoint:
                self.load_last_checkpoint_k_fold(number_fold, total_folds)
            
            train_loader, validation_loader = get_trainloaders(features, targets, number_fold, self.config.batch_size)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=self.optimizer, pct_start=0.1, div_factor=1e3,  max_lr=1e-2, epochs=self.config.n_epochs, steps_per_epoch=len(train_loader))
            
            self.fit(train_loader, validation_loader)
            
            
    def reset(self):
        if self.model_name is None:
            raise Exception("reset call error, the name of the model is not specified!")
            
        self.model = make_model(self.config).to(self.device)
        self.optimizer = self.config.optimizer_class(self.model.parameters(), **self.config.optimizer_params)
        self.criterion = self.config.criterion_class(**self.config.criterion_params) 
        self.epoch = 0
        self.best_summary_score = -10**5
        
        
    def update_dir(self, total_folds, number_fold):
        self.checkpoint_dir = os.path.join(self.current_dir, f'{total_folds}_folds/fold_{number_fold}/checkpoints')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            
        self.log_dir = os.path.join(self.current_dir, f'{total_folds}_folds/fold_{number_fold}/logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.log_path = f'{self.log_dir}/logs.txt'
            
    
    
    #############################################################################################################
    
    def proceed_fit(self): 
        self.fit(self.train_loader, self.validation_loader)
    
    
    #############################################################################################################
    
    def fit(self, train_loader, validation_loader):
        self.train_loader = train_loader
        self.validation_loader = validation_loader
      
        for e in range(self.n_epochs):

            lr = self.optimizer.param_groups[0]['lr']
            timestamp = datetime.utcnow().isoformat()
            self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
         
            summary_loss_train, final_scores_train = self.train_one_epoch(train_loader)

            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss_train.avg:.5f}, final_score: {final_scores_train.avg:.5f}, time: {(time.time() - t):.5f}')
            self.save(f'{self.checkpoint_dir}/last-checkpoint.bin')

            t = time.time()
            summary_loss_val, final_scores_val = self.validation(validation_loader)

            self.log(f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss_val.avg:.5f}, final_score: {final_scores_val.avg:.5f}, time: {(time.time() - t):.5f}')
            
            if final_scores_val.avg > self.best_summary_score:
                self.best_summary_score = final_scores_val.avg
                self.best_summary_loss = summary_loss_val.avg
                self.model.eval()
                self.save(f'{self.checkpoint_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{self.checkpoint_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                    os.remove(path)

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=final_scores_val.avg)

            
            self.epoch += 1
        
        
    def train_one_epoch(self, train_loader):
        self.model.train()
        self.criterion.train()
        
        summary_loss = LossMeter() 
        final_scores = AverageMeter()#LogLossMeter()
        
        t = time.time()
        for step, batch in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )

            targets = batch["target"].to(self.device).float()
            
            tabular_features = batch["tabular_features"].to(self.device).float()
            
            batch_size = tabular_features.shape[0]
            
            self.optimizer.zero_grad()
            outputs = self.model(tabular_features)
            

            loss = self.criterion(outputs, targets)
            loss.backward()
            
            summary_loss(torch.sigmoid(outputs.detach().cpu()), targets.detach().cpu())
            final_scores.update(-loss.detach().item(), batch_size)
            

            self.optimizer.step()

            if self.config.step_scheduler:
                self.scheduler.step()
            
        return summary_loss, final_scores
        
        
    def validation(self, val_loader):
        self.model.eval()
        self.criterion.eval()
        
        summary_loss = LossMeter() 
        final_scores = AverageMeter()
        
        t = time.time()
        for step, batch in enumerate(val_loader):
            
            if step % 10 == 0:
                print(
                    f'Val Step {step}/{len(val_loader)}, ' + \
                    f'summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.5f}, ' + \
                    f'time: {(time.time() - t):.5f}', end='\r'
                    )
            
                
            targets = batch["target"].to(self.device).float()
            tabular_features = batch["tabular_features"].to(self.device).float()
            
            batch_size = tabular_features.shape[0]
                
            with torch.no_grad():
                outputs = self.model(tabular_features)
                
       
            loss = self.criterion(outputs, targets)
        
            summary_loss(torch.sigmoid(outputs.detach().cpu()), targets.detach().cpu())    
            final_scores.update(-loss.detach().item(), batch_size)
            
        return summary_loss, final_scores
        
        
        
    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'best_summary_score': self.best_summary_score,
            'epoch': self.epoch,
        }, path)

        
        
    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.best_summary_score = checkpoint['best_summary_score']
        self.epoch = checkpoint['epoch'] + 1  
        
        
      
    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')
        
 ################################################################################################################  
 
    def load_last_checkpoint(self):
        checkpoint = Path(self.current_dir) / 'checkpoints' / 'last-checkpoint.bin'          #f'{self.checkpoint_dir}/last-checkpoint.bin'
        self.load(checkpoint)
        
    
    def load_last_checkpoint_k_fold(self, fold_number, total_folds=5):
        checkpoint  = Path(self.current_dir) / f'{total_folds}_folds/fold_{fold_number}/checkpoints/last-checkpoint.bin'           
        self.load(checkpoint)
        
 ################################################################################################################       
    
    def load_best_checkpoint(self):
        checkpoint_dir  = Path(self.current_dir) / 'checkpoints' 
        checkpoint = checkpoint_dir / sorted(glob(f'{checkpoint_dir}/best-checkpoint-*epoch.bin'))[-1]
              
        self.load(checkpoint)
    
    
    def load_best_checkpoint_k_fold(self, fold_number, total_folds=5):
        checkpoint_dir  = Path(self.current_dir) / f'{total_folds}_folds/fold_{fold_number}/checkpoints'
        checkpoint = checkpoint_dir / sorted(glob(f'{checkpoint_dir}/best-checkpoint-*epoch.bin'))[-1]
              
        self.load(checkpoint)
    