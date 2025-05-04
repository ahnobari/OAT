from ._models import *
import torch.distributed as dist
import torch

from torch import nn
from torch.nn import functional as F
import numpy as np

import torch.utils
from tqdm.auto import tqdm, trange

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity
import bitsandbytes as bnb
import torch_optimizer as topt
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMScheduler
import os

class NFAE(nn.Module):
    def __init__(self, in_channels=1, resolution=96, z_channels=1, dec_out_channels=128, recon_loss='l1', out_act='tanh'):
        super(NFAE, self).__init__()
        self.in_channels = in_channels
        self.encoder = Encoder(in_channels=in_channels, resolution=resolution, z_channels=z_channels)
        self.decoder = Decoder(in_channels=in_channels, resolution=resolution, z_channels=z_channels, out_ch=in_channels)
        self.renderer = ConcatWrapper(z_dec_channels=dec_out_channels, in_channels=in_channels, out_channels=in_channels, out_act=out_act)
        
        if recon_loss == 'l1':
            self.recon_loss = nn.L1Loss()
        elif recon_loss == 'l2':
            self.recon_loss = nn.MSELoss()
        else:
            self.recon_loss = recon_loss
    
    def forward(self, input_batch, compute_loss=True, shape_up=False, latent_only=False):
        return self.call(input_batch, compute_loss=compute_loss, shape_up=shape_up, latent_only=latent_only)
    
    def call(self, input_batch, compute_loss=True, shape_up=False, latent_only=False):
        
        # # device check
        # if input_batch['inp'].device != self.encoder.in_channels.device:
        
        z = self.encoder(input_batch['inp'])
        
        if latent_only:
            return {'latent': z}
        
        if compute_loss:
            loss = 0.0
            loss_dict = dict()
        
        phi = self.decoder(z)
        
        pred_patch = self.renderer(phi,input_batch['gt_coord'],input_batch['gt_cell'])
        
        if compute_loss:
            if 'mask' in input_batch:
                n_channels = pred_patch.shape[1]
                pred_patch = pred_patch[input_batch['mask']]
                recon_loss = self.recon_loss(pred_patch, input_batch['gt'][input_batch['mask']])
                if shape_up:
                    pred_patch_final = []
                    pred_patch = pred_patch.view(-1)
                    current_index = 0
                    for i in range(input_batch['inp'].shape[0]):
                        size = input_batch['sizes'][i][0] * input_batch['sizes'][i][1] * n_channels
                        pred_patch_final.append(pred_patch[current_index:current_index + size].reshape(n_channels, input_batch['sizes'][i][0], input_batch['sizes'][i][1]))
                    
                        current_index += size
                    pred_patch = pred_patch_final
                    
            else:
                recon_loss = self.recon_loss(pred_patch, input_batch['gt'])
            
            loss_dict['recon_loss'] = recon_loss
            
            loss = recon_loss
        
            loss_dict['loss'] = loss
            
            output = {
                'pred': pred_patch,
                'latent': z
            }
            
            return output, loss_dict
        else:
            if 'mask' in input_batch:
                n_channels = pred_patch.shape[1]
                pred_patch = pred_patch[input_batch['mask']]
                
                if shape_up:
                    pred_patch_final = []
                    pred_patch = pred_patch.view(-1)
                    current_index = 0
                    for i in range(input_batch['inp'].shape[0]):
                        size = input_batch['sizes'][i][0] * input_batch['sizes'][i][1] * n_channels
                        pred_patch_final.append(pred_patch[current_index:current_index + size].reshape(n_channels, input_batch['sizes'][i][0], input_batch['sizes'][i][1]))
                    
                        current_index += size
                    
                    pred_patch = pred_patch_final
            
            output = {
                'pred': pred_patch,
                'latent': z
            }
                
            return output
        
    def load_from_trainer_checkpoint(self, checkpoint_path, strict=True):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            print(f"Loaded model from {checkpoint_path}")
        else:
            print(f"Checkpoint {checkpoint_path} not found.")