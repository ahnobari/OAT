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

def load_OAT_AE(data_path='Dataset', split='train', subset='pre_training', **kawrgs):
    if split == 'test':
        data_path = os.path.join(data_path,'test_data')
    elif subset=='pre_training':
        data_path = os.path.join(data_path,'pre_training')
    elif subset=="labeled":
        data_path = os.path.join(data_path,'labeled_data')
    elif subset=="DOMTopoDiff":
        data_path = os.path.join(data_path,'DOMTopoDiff')
        
    print(f"Loading data from {data_path}")
    
    tops = np.load(os.path.join(data_path,'topologies.npy'), allow_pickle=True)
    shapes = np.load(os.path.join(data_path,'shapes.npy'))
    
    tensors = []
    
    for i in trange(len(tops), desc="Processing tensors"):
        tensors.append(torch.tensor(tops[i].reshape(shapes[i])[None]*255, dtype=torch.uint8))
    
    dataset = AEDataset(tensors=tensors, **kawrgs)
    
    return dataset
    

class AEDataset(Dataset):
    """
    Unified dataset class for Auto Encoder that works with preloaded torch tensors.
    Expects a list of uint8 CHW tensors with values [0, 255]
    """
    
    def __init__(self, 
                 tensors,
                 encoder_res=256,
                 patch_size=32,
                 resize_gt_lb=64,
                 resize_gt_ub=1024,
                 p_whole=0.0,
                 p_max=1.0,
                 square_crop=False,
                 full_sampling=False,
                 square_sampling=False):
        
        self.tensors = tensors  # List of CHW tensors, uint8 [0, 255]
        self.encoder_res = encoder_res
        self.patch_size = patch_size
        self.resize_gt_lb = resize_gt_lb
        self.resize_gt_ub = resize_gt_ub
        self.p_whole = p_whole
        self.p_max = p_max
        self.square_crop = square_crop
        self.full_sampling = full_sampling
        self.square_sampling = square_sampling
        
        # Set up transforms
        self.normalize = transforms.Normalize(0.5, 0.5)
        self.resize = transforms.Resize
        
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
    
    @staticmethod
    def _collate_inf(batch):
        input_images = torch.stack([b['inp'] for b in batch])
        
        out = {
            'inp': input_images
        }
        
        return out
    
    @staticmethod
    def _collate_fn(batch):
        
        input_images = torch.stack([b['inp'] for b in batch])
        
        # check if all tensors are the same size
        multi_size = False
        max_shape = [batch[0]['gt'].shape[1], batch[0]['gt'].shape[2]]
        sizes = []
        for b in batch:
            if not multi_size:
                if b['gt'].shape[1] != max_shape[0] or b['gt'].shape[2] != max_shape[1]:
                    multi_size = True
            
            if b['gt'].shape[1] > max_shape[0]:
                max_shape[0] = b['gt'].shape[1]
            if b['gt'].shape[2] > max_shape[1]:
                max_shape[1] = b['gt'].shape[2]
            sizes.append((b['gt'].shape[1], b['gt'].shape[2]))
        
        if multi_size:
            # pad all tensors to the max size and make a mask
            gt_images = torch.zeros(len(batch), 1, max_shape[0], max_shape[1])
            gt_coords = torch.zeros(len(batch), max_shape[0], max_shape[1], 2)
            gt_cells = torch.zeros(len(batch), max_shape[0], max_shape[1], 2)
            masks = torch.zeros(len(batch), 1, max_shape[0], max_shape[1], dtype=torch.bool)
            for i, b in enumerate(batch):
                gt_images[i, :, :b['gt'].shape[1], :b['gt'].shape[2]] = b['gt']
                gt_coords[i, :b['gt_coord'].shape[0], :b['gt_coord'].shape[1], :] = b['gt_coord']
                gt_cells[i, :b['gt_cell'].shape[0], :b['gt_cell'].shape[1], :] = b['gt_cell']
                masks[i, :, :b['gt'].shape[1], :b['gt'].shape[2]] = True
                
            out = {
                'inp': input_images,
                'gt': gt_images,
                'gt_coord': gt_coords,
                'gt_cell': gt_cells,
                'mask': masks,
                'sizes': torch.tensor(sizes, dtype=torch.int32)
            }
            
        else:
            gt_images = torch.stack([b['gt'] for b in batch])
            gt_coords = torch.stack([b['gt_coord'] for b in batch])
            gt_cells = torch.stack([b['gt_cell'] for b in batch])
        
            out = {
                'inp': input_images,
                'gt': gt_images,
                'gt_coord': gt_coords,
                'gt_cell': gt_cells
            }
            
        return BatchDict(out)
            
    def __getitem__(self, idx):
        # Get image and optionally crop to square
        img = self.tensors[idx]  # CHW format, uint8
        if self.square_crop:
            img = self._center_crop_square(img)
            pad_info = (0, 0, 0, 0)
        else:
            img,pad_info = self._center_pad_square(img)
            
        orig_size = img.shape[-1]  # Assuming square after crop
        
        # Create input image at fixed encoder resolution
        resize_transform = self.resize(self.encoder_res, antialias=True)
        inp = resize_transform(img.unsqueeze(0)).squeeze(0)
        inp = self._preprocess_tensor(inp)
        
        # Determine ground truth resolution
        if self.full_sampling:
            gt_res = orig_size
        elif random.random() < self.p_whole:
            gt_res = self.patch_size
        elif random.random() < self.p_max:
            gt_res = min(orig_size, self.resize_gt_ub)
        else:
            gt_res = random.randint(self.resize_gt_lb, 
                                  min(orig_size, self.resize_gt_ub))
        
        # Resize ground truth
        if not self.full_sampling:
            resize_transform = self.resize(gt_res, antialias=True)
            gt = resize_transform(img.unsqueeze(0)).squeeze(0)
            gt = self._preprocess_tensor(gt)
        else:
            gt = self._preprocess_tensor(img)
        
        # Random crop patch from ground truth
        if self.full_sampling and not self.square_sampling:
            start_h = pad_info[0]
            start_w = pad_info[2]
            end_h = gt.shape[1] - pad_info[1]
            end_w = gt.shape[2] - pad_info[3]
            
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
        elif self.full_sampling and self.square_sampling:
            start_h = 0
            start_w = 0
            end_h = gt.shape[1]
            end_w = gt.shape[2]
            
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
        elif gt_res > self.patch_size:
            
            padded_ratio_h = pad_info[0] / img.shape[-1]
            padded_ratio_w = pad_info[2] / img.shape[-1]
            
            min_start_h = int(gt_res * padded_ratio_h)
            min_start_w = int(gt_res * padded_ratio_w)
            max_start_h = int(gt_res * (1 - padded_ratio_h) - self.patch_size)
            max_start_w = int(gt_res * (1 - padded_ratio_w) - self.patch_size)
            
            if max_start_h < min_start_h:
                min_start_h = max_start_h = gt_res // 2 - self.patch_size // 2
            if max_start_w < min_start_w:
                min_start_w = max_start_w = gt_res // 2 - self.patch_size // 2
            
            start_h = random.randint(min_start_h, max_start_h)
            start_w = random.randint(min_start_w, max_start_w)
            gt = gt[:, start_h:start_h + self.patch_size, 
                      start_w:start_w + self.patch_size]
                      
            # Calculate relative coordinates for the patch
            rel_start_h = start_h / gt_res
            rel_start_w = start_w / gt_res
            rel_end_h = (start_h + self.patch_size) / gt_res
            rel_end_w = (start_w + self.patch_size) / gt_res
            
            # bring to -1 to 1 range
            rel_start_h = 2 * rel_start_h - 1
            rel_start_w = 2 * rel_start_w - 1
            rel_end_h = 2 * rel_end_h - 1
            rel_end_w = 2 * rel_end_w - 1
            
            # Generate coordinate and cell grids for the patch
            coord, cell = make_coord_cell_grid(
                self.patch_size,
                range=[[rel_start_w, rel_end_w], 
                      [rel_start_h, rel_end_h]]
            )
            
            cell[:] = torch.tensor([2/gt_res, 2/gt_res])
        else:
            coord, cell = make_coord_cell_grid(self.patch_size)
        
        out = {
            'inp': inp,                    # C×H×W normalized
            'gt': gt,                      # C×P×P normalized
            'gt_coord': coord,             # P×P×2
            'gt_cell': cell,               # P×P×2
        }
        
        return out

    def visualize(self, idx=None):
        if idx is None:
            idx = random.randint(0, len(self) - 1)
        samples = self.__getitem__(idx)
        # Set up the figure
        fig = plt.figure(figsize=(20, 10))

        # 1. Original Input Image
        ax = plt.subplot(1, 5, 1)
        img = self.tensors[idx]
        # Convert from [-1,1] to [0,1] for visualization
        plt.imshow(img.permute(1, 2, 0).numpy(), cmap='Grays')
        ax.set_title('Original Input')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        # 2. Encoder Input
        ax = plt.subplot(1, 5, 2)
        inp_viz = (samples['inp'] + 1) / 2
        plt.imshow(inp_viz.permute(1, 2, 0).numpy(), cmap='Grays')
        ax.set_title(f'Encoder Input ({self.encoder_res}x{self.encoder_res})')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        # 3. GT Patch
        ax = plt.subplot(1, 5, 3)
        patch = samples['gt']
        plt.imshow(patch.permute(1, 2, 0).numpy(), cmap='Grays')
        ax.set_title('Random GT Patch')
        ax.axis('off')

        # 4. Coordinate Grid Visualization
        ax = plt.subplot(1, 5, 4)
        coord = (samples['gt_coord'].reshape(-1, 2) + 1 )/2
        sizes = samples['gt_cell'].reshape(-1, 2)/2
        patch_vals = samples['gt'].reshape(-1)
        plt.imshow(inp_viz.permute(1, 2, 0).numpy(), cmap='Grays')
        # draw square around each cell
        for i in range(len(coord)):
            x, y = coord[i] * self.encoder_res - sizes[i][0]* self.encoder_res/2
            s = sizes[i] * self.encoder_res
            face_alpha = f"#0000ff{int((patch_vals[i].numpy()+1)/2*255):02x}"
            rect = plt.Rectangle((x-s[0]/2, y-s[1]/2), s[0], s[1], linewidth=1, edgecolor='r', facecolor=face_alpha)
            ax.add_patch(rect)

        plt.tight_layout()