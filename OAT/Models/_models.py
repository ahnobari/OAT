import torch
import torch.nn as nn
from ._net import Encoder, Decoder, ConcatWrapper, Discriminator, VectorQuantizer
import torch.distributed as dist
import os

class ResFreeAutoEncoder(nn.Module):
    def __init__(self, in_channels=1, ae_resolution=96, z_channels=1, tanh_output=True, use_quant = False, use_gan = False, l1_weight=1.0, quant_weight=1.0, gan_loss_weight=None):
        super(ResFreeAutoEncoder, self).__init__()
        self.in_channels = in_channels
        self.encoder = Encoder(in_channels=in_channels, resolution=ae_resolution, z_channels=z_channels)
        self.decoder = Decoder(in_channels=in_channels, resolution=ae_resolution, z_channels=z_channels, out_ch=in_channels)
        self.renderer = ConcatWrapper(z_dec_channels=128, in_channels=in_channels, out_channels=in_channels)
        self.discriminator = Discriminator(input_nc=in_channels)
        self.quantizer = VectorQuantizer(e_dim=z_channels)
        
        self.l1_weight = l1_weight
        self.quant_weight = quant_weight
        self.gan_loss_weight = gan_loss_weight
        self.use_quant = use_quant
        self.use_gan = use_gan
        self.tanh_output = tanh_output
        
    def encode(self, input_batch):
        
        z = self.encoder(input_batch['inp'])

        n_patches = input_batch['gt'].shape[1]

        n_dims = len(z.size())
        expand_dims = [-1] * (n_dims+1)
        expand_dims[1] = n_patches
        
        enc_out = z.unsqueeze(1).expand(torch.Size(expand_dims))
        
        return enc_out
    
    def decode(self, input_batch, z):
        
        n_patches = input_batch['gt'].shape[1]
        z = z.reshape(-1, *z.shape[2:])
        
        phi = self.decoder(z)
        
        coords = input_batch['gt_coord']
        coords = coords.reshape(-1, *coords.shape[2:])
        
        cell = input_batch['gt_cell']
        cell = cell.reshape(-1, *cell.shape[2:])
        
        pred_patch = self.renderer(phi, coords, cell)
        pred_patch = pred_patch.reshape(-1, n_patches, *pred_patch.shape[1:])
        
        if self.tanh_output:
            pred_patch = torch.tanh(pred_patch)
        
        return pred_patch
    
    def recon(self, input_batch, compute_loss=True):
        enc_out = self.encode(input_batch)

        if self.use_quant:
            z, quant_loss, _ = self.quantizer(enc_out.reshape(-1, *enc_out.shape[2:]))
            z = z.reshape(enc_out.shape)
            loss += quant_loss * self.quant_weight
        else:
            z = enc_out

        pred_patch = self.decode(input_batch, z)
        
        if compute_loss:
            loss = {}
            l1_loss = torch.abs(pred_patch - input_batch['gt']).mean()
            loss['recon_loss'] = l1_loss
            loss['loss'] = l1_loss * self.l1_weight
            if self.use_quant:
                loss['quant_loss'] = quant_loss 
                loss['loss'] += loss['quant_loss'] * self.quant_weight
            
            if self.use_gan:
                disc_out = self.discriminator(pred_patch.reshape(-1, *pred_patch.shape[2:]))
                disc_loss = -torch.mean(disc_out)

                if self.gan_loss_weight is not None:
                    g_weight = self.gan_loss_weight
                else:
                    nll_loss = l1_loss * self.l1_weight
                    g_weight = self.calculate_adaptive_g_w(nll_loss, disc_loss, self.renderer.get_last_layer_weight())
                
                loss['gan_loss'] = disc_loss
                loss['loss'] += g_weight * disc_loss

            return pred_patch, loss
        else:
            return pred_patch
        
    def loss(self, input_batch):
        pred_patch, loss = self.recon(input_batch)
        return loss
    
    def forward(self, input_batch, compute_loss=True):
        out = self.recon(input_batch, compute_loss=compute_loss)
        return out
    
    def calculate_adaptive_g_w(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        if world_size > 1:
            dist.all_reduce(nll_grads, op=dist.ReduceOp.SUM)
            nll_grads.div_(world_size)
            dist.all_reduce(g_grads, op=dist.ReduceOp.SUM)
            g_grads.div_(world_size)
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight