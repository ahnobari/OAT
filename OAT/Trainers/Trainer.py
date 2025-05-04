import torch.distributed as dist
import torch

from torch import nn
import numpy as np

import torch.utils
from tqdm.auto import tqdm

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import bitsandbytes as bnb
import torch_optimizer as topt
import os

class Trainer:
    def __init__(self, model: nn.Module, lr: float = 1e-4, weight_decay: float = 1e-4,
                 cosine_schedule: bool = True, lr_final: float = 1e-5, schedule_max_steps: int = 100,
                 warmup_steps: int = 200, device: str = None, multi_gpu: bool = False, mixed_precision: bool = True,
                 DDP_train: bool = True, Compile: bool = True, checkpoint_path: str = None,
                 optimizer: str = 'AdamW'):
        
        self.multi_gpu = multi_gpu
        self.DDP = DDP_train if multi_gpu else False
        self.mixed_precision = mixed_precision
        self.optimizer = optimizer
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        self.model = model
        
        if hasattr(self.model, 'compile') and Compile:
            self.model.compile()
        
        if self.DDP:
            self.setup_ddp()
        elif self.multi_gpu and type(self.multi_gpu) is list:
            self.model = self.model.to(self.device)
            self.model = nn.DataParallel(self.model, device_ids=multi_gpu)
        elif self.multi_gpu:
            self.model = self.model.to(self.device)
            self.model = nn.DataParallel(self.model)
        else:
            self.model = self.model.to(self.device)
        
        self.lr = lr
        self.weight_decay = weight_decay
        
        param_list = self.model.parameters()
            
        
        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(param_list, lr=lr, weight_decay=weight_decay)
        elif optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(param_list, lr=lr, weight_decay=weight_decay)
        elif optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(param_list, lr=lr, weight_decay=weight_decay)
        elif optimizer == 'Adam8':
            self.optimizer = bnb.optim.Adam8bit(param_list, lr=lr, weight_decay=weight_decay)
        elif optimizer == 'Adafactor':
            self.optimizer = topt.Adafactor(param_list, lr=lr, weight_decay=weight_decay)
        
        self.cosine_schedule = cosine_schedule
        self.lr_final = lr_final
        self.schedule_max_steps = schedule_max_steps
        
        if self.cosine_schedule:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=schedule_max_steps, eta_min=lr_final)
        else:
            self.scheduler = None
        
        self.current_epoch = 0
        
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

        torch.cuda.empty_cache()

    def setup_ddp(self):
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '12355'
        
        self.rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        dist.init_process_group(backend='nccl')

        torch.cuda.set_device(self.rank)

        self.model = self.model.to(self.rank)
        
        self.model = DDP(self.model, device_ids=[self.rank],find_unused_parameters=True)

    def cleanup_ddp(self):
        if self.DDP:
            dist.destroy_process_group()

    def is_main_process(self):
        return self.rank == 0 if self.DDP else True

    def save_checkpoint(self, path):
        if self.is_main_process():
            checkpoint = {
                'model_state_dict': self.model.module.state_dict() if isinstance(self.model, (nn.DataParallel, DDP)) else self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'current_epoch': self.current_epoch,
            }
            if self.scheduler:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            torch.save(checkpoint, path)
    
    def load_checkpoint(self, path, model_only=False):
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(self.model, (nn.DataParallel, DDP)):
            self.model.module.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        if not model_only:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except:
                if self.is_main_process():
                    print("Optimizer state dict not found in checkpoint or incompatible with current optimizer.")

            self.current_epoch = checkpoint['current_epoch']

            try:
                if self.scheduler and 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except:
                if self.is_main_process():
                    print("Scheduler state dict not found in checkpoint or incompatible with current scheduler.")
            
    def reset_optimizer(self):
        param_list = self.model.parameters()
        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(param_list, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(param_list, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(param_list, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'Adam8':
            self.optimizer = bnb.optim.Adam8bit(param_list, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'Adafactor':
            self.optimizer = topt.Adafactor(param_list, lr=self.lr, weight_decay=self.weight_decay)
        
        if self.cosine_schedule:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.schedule_max_steps, eta_min=self.lr_final)
        else:
            self.scheduler = None
            
    def train(self, dataset, data_idx, batch_size, epochs=100, continue_loop=True, verbose=True, checkpoint_interval=10, checkpoint_dir='Checkpoints', **kwargs):
        
        if not continue_loop:
            self.model.train()
            self.current_epoch = 0
            self.reset_optimizer()
            
        if self.mixed_precision:
            scaler = torch.amp.GradScaler("cuda")
        
        torch.cuda.empty_cache()
        
        if self.DDP:
            data_idx = np.array_split(data_idx, self.world_size)[self.rank]
        
        steps_per_epoch = int(np.ceil(len(data_idx) / batch_size))
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(epochs - self.current_epoch):
            
            collate_fn = dataset._collate_fn if hasattr(dataset, '_collate_fn') else None
            loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, data_idx), batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True, collate_fn=collate_fn)

            if verbose and self.is_main_process():
                prog = tqdm(loader, total=len(loader))
            else:
                prog = loader

            epoch_loss = 0
                        
            for i, batch in enumerate(prog):
                self.optimizer.zero_grad()
                inputs_batch = batch.to(self.device)
                
                if self.mixed_precision:
                    with torch.amp.autocast("cuda"):
                        if self.multi_gpu:
                            loss_dict = self.model(inputs_batch, compute_loss=True)[1]
                        else:
                            loss_dict = self.model(inputs_batch, compute_loss=True)[1]
                        loss = loss_dict['loss']
                else:
                    if self.multi_gpu:
                        loss_dict = self.model(inputs_batch, compute_loss=True)[1]
                    else:
                        loss_dict = self.model(inputs_batch, compute_loss=True)[1]
                    loss = loss_dict['loss']
                    
                if self.mixed_precision:
                    scaler.scale(loss).backward()
                    
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.grad.data.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                    
                    
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                
                epoch_loss += loss_dict['loss'].item()
                
                current_learning_rate = self.optimizer.param_groups[0]['lr']
                
                if self.is_main_process() and verbose:
                    post_fix_str = f'Epoch {self.current_epoch}, Epoch Loss: {epoch_loss/(i+1):.5f}, LR: {current_learning_rate:.5f}'
                    
                    for key, value in loss_dict.items():
                        post_fix_str += f', {key}: {value.item():.5f}'
                        
                    prog.set_postfix_str(post_fix_str)
                    
            if self.cosine_schedule:
                curent_step = self.current_epoch
                if curent_step <= self.schedule_max_steps:
                    self.scheduler.step()
                    
            self.current_epoch += 1
            
            if verbose and self.is_main_process():
                print(f'Epoch {self.current_epoch}, Loss: {epoch_loss/steps_per_epoch}')
                
            self.save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_epoch_{self.current_epoch}.pth'))

            if (self.current_epoch-1) % checkpoint_interval == 0:
                pass
            elif self.is_main_process():
                os.remove(os.path.join(checkpoint_dir, f'checkpoint_epoch_{self.current_epoch-1}.pth'))

        if self.DDP:
            dist.barrier()
            
    def compute_latents(self, dataset, data_idx, batch_size, verbose=True):
        self.model.eval()
        
        if self.DDP:
            if self.is_main_process():
                data_idx = np.array_split(data_idx, self.world_size)
                sizes = [len(idx) for idx in data_idx]
                data_idx = data_idx[self.rank]
            else:
                data_idx = np.array_split(data_idx, self.world_size)[self.rank]
        
        collate_fn = dataset._collate_fn if hasattr(dataset, '_collate_fn') else None
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, data_idx), batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False, collate_fn=collate_fn)
        
        all_latents = []
        
        if verbose and self.is_main_process():
            prog = tqdm(loader, total=len(loader))
        else:
            prog = loader
        
        for i, batch in enumerate(prog):
            if self.multi_gpu and self.DDP or not self.multi_gpu:
                inputs_batch = batch.to(self.device)
            else:
                inputs_batch = batch
            with torch.no_grad():
                if self.DDP:
                    with self.model.no_sync():
                        if self.mixed_precision:
                            with torch.amp.autocast("cuda"):
                                latents = self.model(inputs_batch, latent_only=True)['latent']
                        else:
                            latents = self.model(inputs_batch, latent_only=True)['latent']
                else:
                    if self.mixed_precision:
                        with torch.amp.autocast("cuda"):
                            latents = self.model(inputs_batch, latent_only=True)['latent']
                    else:
                        latents = self.model(inputs_batch, latent_only=True)['latent']
                
                all_latents.append(latents.detach().cpu())
                
        all_latents = torch.cat(all_latents, dim=0)
        self.model.train()
        return all_latents