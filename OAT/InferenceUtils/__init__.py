import torch
from typing import Callable
from ..Models.NFAE import NFAE

def INFD_Latent_Map(latent : torch.Tensor,
                    AE : NFAE,
                    coords : torch.Tensor,
                    cells : torch.Tensor,
                    inverse_normalize : Callable = None,
                    clip_to_range : bool = True,
                    clip_range : tuple = (-1, 1)):
    
    if inverse_normalize is not None:
        if clip_to_range:
            latent = torch.clamp(latent, clip_range[0], clip_range[1])
        latent = inverse_normalize(latent)
    
    
    # 1. Get the latent map
    pred_patch = AE.decoder(latent)
    pred_patch = AE.renderer(pred_patch,coords,cells)

    return pred_patch