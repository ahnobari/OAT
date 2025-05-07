import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from tqdm.auto import trange
from ._utils import *

def load_all(data_path, split):
    if split == 'test':
        domtopodiff_tops = np.load(os.path.join(data_path,'DOMTopoDiff', 'Test', 'topologies.npy'), allow_pickle=True)
        domtopodiff_shapes = np.load(os.path.join(data_path,'DOMTopoDiff', 'Test', 'shapes.npy'))
        domtopodiff_bcs = np.load(os.path.join(data_path,'DOMTopoDiff', 'Test', 'bcs.npy'), allow_pickle=True)
        domtopodiff_vfs = np.load(os.path.join(data_path,'DOMTopoDiff', 'Test', 'vfs.npy'), allow_pickle=True)
        domtopodiff_loads = np.load(os.path.join(data_path,'DOMTopoDiff', 'Test', 'loads.npy'), allow_pickle=True).astype(np.object_)
        
        labeled_tops = np.load(os.path.join(data_path,'test_data', 'topologies.npy'), allow_pickle=True)
        labeled_shapes = np.load(os.path.join(data_path,'test_data', 'shapes.npy'))
        labeled_bcs = np.load(os.path.join(data_path,'test_data', 'bcs.npy'), allow_pickle=True)
        labeled_vfs = np.load(os.path.join(data_path,'test_data', 'vfs.npy'), allow_pickle=True)
        labeled_loads = np.load(os.path.join(data_path,'test_data', 'loads.npy'), allow_pickle=True)
        
        tops = np.concatenate([labeled_tops, domtopodiff_tops], axis=0)
        shapes = np.concatenate([labeled_shapes, domtopodiff_shapes], axis=0)
        bcs = np.concatenate([labeled_bcs, domtopodiff_bcs], axis=0)
        vfs = np.concatenate([labeled_vfs, domtopodiff_vfs], axis=0)
        loads = np.array([labeled_loads[i] for i in range(len(labeled_loads))] +
                         [domtopodiff_loads[i] for i in range(len(domtopodiff_loads))], dtype=object)
        
    else:
        domtopodiff_tops = np.load(os.path.join(data_path,'DOMTopoDiff', 'topologies.npy'), allow_pickle=True)
        domtopodiff_shapes = np.load(os.path.join(data_path,'DOMTopoDiff', 'shapes.npy'))
        domtopodiff_bcs = np.load(os.path.join(data_path,'DOMTopoDiff', 'bcs.npy'), allow_pickle=True)
        domtopodiff_vfs = np.load(os.path.join(data_path,'DOMTopoDiff', 'vfs.npy'), allow_pickle=True)
        domtopodiff_loads = np.load(os.path.join(data_path,'DOMTopoDiff', 'loads.npy'), allow_pickle=True)
        
        labeled_tops = np.load(os.path.join(data_path,'labeled_data', 'topologies.npy'), allow_pickle=True)
        labeled_shapes = np.load(os.path.join(data_path,'labeled_data', 'shapes.npy'))
        labeled_bcs = np.load(os.path.join(data_path,'labeled_data', 'bcs.npy'), allow_pickle=True)
        labeled_vfs = np.load(os.path.join(data_path,'labeled_data', 'vfs.npy'), allow_pickle=True)
        labeled_loads = np.load(os.path.join(data_path,'labeled_data', 'loads.npy'), allow_pickle=True)
        
        tops = np.concatenate([labeled_tops, domtopodiff_tops], axis=0)
        shapes = np.concatenate([labeled_shapes, domtopodiff_shapes], axis=0)
        bcs = np.concatenate([labeled_bcs, domtopodiff_bcs], axis=0)
        vfs = np.concatenate([labeled_vfs, domtopodiff_vfs], axis=0)
        loads = np.array([labeled_loads[i] for i in range(len(labeled_loads))] +
                         [domtopodiff_loads[i] for i in range(len(domtopodiff_loads))], dtype=object)
    
    return tops, shapes, bcs, vfs, loads
        

def load_OAT_CDiff(latents_path, data_path='Dataset', split='train', subset='labeled', ignore_BC=False, ignore_vf=False, cell_size=True, **kawrgs):
    if split == 'test':
        if subset == 'DOMTopoDiff':
            data_path = os.path.join(data_path,'DOMTopoDiff', 'Test')
        else:
            data_path = os.path.join(data_path,'test_data')
    elif subset=="labeled":
        data_path = os.path.join(data_path,'labeled_data')
    elif subset=="DOMTopoDiff":
        data_path = os.path.join(data_path,'DOMTopoDiff')
        
    print(f"Loading data from {data_path}")
    
    if subset == 'all':
        tops, shapes, bcs, vfs, loads = load_all(data_path, split)
    else:
        tops = np.load(os.path.join(data_path,'topologies.npy'), allow_pickle=True)
        shapes = np.load(os.path.join(data_path,'shapes.npy'))
        bcs = np.load(os.path.join(data_path,'bcs.npy'), allow_pickle=True)
        vfs = np.load(os.path.join(data_path,'vfs.npy'), allow_pickle=True)
        loads = np.load(os.path.join(data_path,'loads.npy'), allow_pickle=True)
    
    tensors = []
    
    for i in trange(len(tops), desc="Processing tensors"):
        tensors.append(torch.tensor(tops[i].reshape(shapes[i])[None]*255, dtype=torch.uint8))
    
    # dataset = AEDataset(tensors=tensors, **kawrgs)
    if latents_path is not None:
        latents = torch.load(latents_path)
    else:
        latents = None
    
    if cell_size:
        Cs = [vfs, shapes/shapes.max(1)[:, None], 1/shapes.max(1)[:, None]] if not ignore_vf else [shapes/shapes.max(1)[:, None], 1/shapes.max(1)[:, None]]
    else:
        Cs = [vfs, shapes/shapes.max(1)[:, None]] if not ignore_vf else [shapes/shapes.max(1)[:, None]]
    
    dataset = DiffDataset(
        tensors=tensors,
        latent_tensors=latents,
        BCs=[bcs,loads] if not ignore_BC else [],
        Cs=Cs,
        **kawrgs
    )
    
    return dataset

class DiffDataset(Dataset):
    """
    Diffusion Dataset for Latent Diffusion (Conditional and Unconditional)
    """
    
    def __init__(self, 
                 tensors,
                 latent_tensors,
                 BCs = [],
                 Cs = [],
                 unconditional_prob=0.0,
                 BC_dropout_prob=0.0,
                 C_dropout_prob=0.0,
                 shift = None,
                 scale = None
                 ):
        
        self.tensors = tensors  # List of CHW tensors, uint8 [0, 255]
        self.latent_tensors = latent_tensors
        self.BCs = BCs
        self.Cs = Cs
        self.unconditional_prob = unconditional_prob
        self.BC_dropout_prob = BC_dropout_prob
        self.C_dropout_prob = C_dropout_prob
        
        self.n_BC = len(BCs)
        self.n_C = len(Cs)
        
        self.max_size = 0
        
        if latent_tensors is not None:
            if scale is None:
                scale = latent_tensors.max() - latent_tensors.min()
            if shift is None:
                min_val = latent_tensors.min()
            else:
                min_val = -shift
        else:
            if scale is None:
                scale = 1.0
            if shift is None:
                min_val = 0.0
            else:
                min_val = -shift
        
        # Set up transforms
        self.normalize = transforms.Normalize(0.5, 0.5)
        self.latent_normalize = lambda x: x.add_(-min_val).div_(scale).mul_(2).add_(-1)
        self.resize = transforms.Resize
        
        self.shift = -min_val
        self.scale = scale

        self.inverse_normalize = lambda x: (x + 1)/2 * scale + min_val
        
        if self.n_BC ==0 and self.n_C == 0:
            self.is_unconditional = True
        else:
            self.is_unconditional = False
        
    def __len__(self):
        return len(self.tensors)
    
    def _center_crop_square(self, img):
        """Center crop tensor to square"""
        _, h, w = img.shape
        size = min(h, w)
        top = (h - size) // 2
        left = (w - size) // 2
        return img[:, top:top + size, left:left + size]
    
    def _center_pad_square(self, img, fill=0):
        """Center pad tensor to square"""
        _, h, w = img.shape
        size = max(h, w)
        pad_h = (size - h) // 2
        pad_w = (size - w) // 2
        pad_h2 = size - h - pad_h
        pad_w2 = size - w - pad_w
        
        return torch.nn.functional.pad(img, (pad_w, pad_w2, pad_h, pad_h2), value=fill), (pad_h, pad_h2, pad_w, pad_w2)
    
    def _preprocess_tensor(self, img):
        """Convert uint8 tensor to float and normalize to [-1,1]"""
        img = img.float() / 255.0  # [0,1]
        img = self.normalize(img)  # [-1,1]
        return img
    
    def _collate_fn(self, batch):
        if self.latent_tensors is not None:  
            input_images = torch.stack([b['input_img'] for b in batch])
        Cs = [torch.stack([torch.tensor(b['Cs'][i].astype(np.float32)) for b in batch]) for i in range(len(batch[0]['Cs']))]
        for i in range(len(Cs)):
            if Cs[i].dim() == 1:
                Cs[i] = Cs[i].unsqueeze(1)
        
        if self.n_BC > 0:
            BCs = [torch.cat([torch.tensor(b['BCs'][i].astype(np.float32)) for b in batch]) for i in range(len(batch[0]['BCs']))]
            BC_batch = [torch.cat([torch.tensor(np.repeat(j,b['sizes'][i])).long() for j,b in enumerate(batch)]) for i in range(len(batch[0]['BCs']))]
        else:
            BCs = None
            BC_batch = None
            
        if self.latent_tensors is not None:
            out =  {
                'sample': self.latent_normalize(input_images),
                'Cs': Cs,
                'BCs': BCs,
                'BC_Batch': BC_batch,
                'unconditioned': np.random.rand() < self.unconditional_prob
            }
        else:
            out = {
                'Cs': Cs,
                'BCs': BCs,
                'BC_Batch': BC_batch,
                'unconditioned': np.random.rand() < self.unconditional_prob
            }
        
        return BatchDict(out)
    
    def __getitem__(self, idx):
        if self.latent_tensors is None:
            input_img = None
        else:
            input_img = self.latent_tensors[idx]
        Cs = []
        for i in range(self.n_C):
            Cs.append(np.array(self.Cs[i][idx]))
            if np.random.rand() < self.C_dropout_prob:
                Cs[i] = np.zeros_like(Cs[i]) - 1.0
        
        BCs = []
        sizes = []
        for i in range(self.n_BC):
            BCs.append(np.array(self.BCs[i][idx]))
            sizes.append(self.BCs[i][idx].shape[0])
            
            if np.random.rand() < self.BC_dropout_prob:
                BC_size = list(BCs[i].shape)
                BC_size[0] = 1
                BCs[i] = np.zeros(BC_size) - 1.0
                sizes[i] = 1
        
        return {
            'input_img': input_img,
            'Cs': Cs,
            'BCs': BCs,
            'sizes': sizes
        }
        
    def get_AE_grid(self, idx, batch_size=1):
        # Get image and optionally crop to square
        img = self.tensors[idx]
        shape = self.tensors[idx].shape
        img,pad_info = self._center_pad_square(img)
            
        orig_size = img.shape[-1]  # Assuming square after crop
        gt_res = orig_size

        gt = self._preprocess_tensor(img)
        
        start_h = pad_info[0]
        start_w = pad_info[2]
        end_h = gt.shape[1] - pad_info[1]
        end_w = gt.shape[2] - pad_info[3]
        
        gt_full = gt.clone()
        gt = gt[:, start_h:end_h, start_w:end_w]
                    
        # Calculate relative coordinates for the patch
        rel_start_h = start_h / gt_res
        rel_start_w = start_w / gt_res
        rel_end_h = end_h / gt_res
        rel_end_w = end_w / gt_res
        
        # bring to -1 to 1 range
        rel_start_h = 2 * rel_start_h - 1
        rel_start_w = 2 * rel_start_w - 1
        rel_end_h = 2 * rel_end_h - 1
        rel_end_w = 2 * rel_end_w - 1
        
        # Generate coordinate and cell grids for the patch
        coord, cell = make_coord_cell_grid(
            (end_h - start_h, end_w - start_w),
            range=[[rel_start_w, rel_end_w], 
                    [rel_start_h, rel_end_h]]
        )
        
        cell[:] = torch.tensor([2/gt_res, 2/gt_res])
        
        full_start_h = 0
        full_start_w = 0
        full_end_h = gt_full.shape[1]
        full_end_w = gt_full.shape[2]
        rel_start_h = full_start_h / gt_res
        rel_start_w = full_start_w / gt_res
        rel_end_h = full_end_h / gt_res
        rel_end_w = full_end_w / gt_res
        
        rel_start_h = 2 * rel_start_h - 1
        rel_start_w = 2 * rel_start_w - 1
        rel_end_h = 2 * rel_end_h - 1
        rel_end_w = 2 * rel_end_w - 1
        
        full_coord, full_cell = make_coord_cell_grid(
            (full_end_h - full_start_h, full_end_w - full_start_w),
            range=[[rel_start_w, rel_end_w], 
                    [rel_start_h, rel_end_h]]
        )
        
        full_cell[:] = torch.tensor([2/gt_res, 2/gt_res])
        zero_mask = torch.zeros_like(gt_full)
        zero_mask[:, start_h:end_h, start_w:end_w] = 1.0
        
        out = {
            'gt': gt.unsqueeze(0).expand(batch_size, -1, -1, -1),                      # BxC×P×P
            'gt_coord': coord.unsqueeze(0).expand(batch_size, -1, -1, -1),             # B×P×P×2
            'gt_cell': cell.unsqueeze(0).expand(batch_size, -1, -1, -1),               # B×P×P×2
            'gt_full': gt_full.unsqueeze(0).expand(batch_size, -1, -1, -1),            # BxC×H×W
            'gt_full_coord': full_coord.unsqueeze(0).expand(batch_size, -1, -1, -1),   # B×H×W×2
            'gt_full_cell': full_cell.unsqueeze(0).expand(batch_size, -1, -1, -1),     # B×H×W×2
            'full_zero_mask': zero_mask.unsqueeze(0).expand(batch_size, -1, -1, -1),   # B×C×H×W
        }
        
        return BatchDict(out)