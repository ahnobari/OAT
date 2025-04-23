import os
import random
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from functools import partial
from ._utils import make_coord_cell_grid, get_center_pad_info
from tqdm.auto import trange
import matplotlib.pyplot as plt

def load_OAT_AE(data_path='dataset', split='train', subset='pre_training', **kawrgs):
    if split == 'test':
        data_path = os.path.join(data_path,'test_data')
    elif subset=='pre_training':
        data_path = os.path.join(data_path,'pre_training')
    elif subset=="labeled":
        data_path = os.path.join(data_path,'labeled_data')
    
    print(f"Loading data from {data_path}")
    
    tops = np.load(os.path.join(data_path,'topologies.npy'), allow_pickle=True)
    shapes = np.load(os.path.join(data_path,'shapes.npy'))
    
    tensors = []
    
    for i in trange(len(tops), desc="Processing tensors"):
        tensors.append(torch.tensor(tops[i].reshape(shapes[i])[None]*255, dtype=torch.uint8))
    
    dataset = UnifiedINFDPreLoadedDataset(tensors=tensors,**kawrgs)
    
    return dataset
    
class UnifiedINFDPreLoadedDataset(Dataset):
    """
    Unified dataset class for INFD that works with preloaded torch tensors.
    Expects a list of uint8 CHW tensors with values [0, 255]
    """
    
    def __init__(self, 
                 tensors,
                 encoder_res=256,
                 patch_size=32,
                 resize_gt_lb=256,
                 resize_gt_ub=1024,
                 max_downscale=2,
                 p_whole=0.0,
                 p_max=0.0,
                 square_crop=False,
                 full_sampling=False,
                 n_patches=4):
        
        self.tensors = tensors  # List of CHW tensors, uint8 [0, 255]
        self.encoder_res = encoder_res
        self.patch_size = patch_size
        self.resize_gt_lb = resize_gt_lb
        self.resize_gt_ub = resize_gt_ub
        self.p_whole = p_whole
        self.p_max = p_max
        self.square_crop = square_crop
        self.full_sampling = full_sampling
        self.n_patches = n_patches
        self.max_downscale = max_downscale
        
        if self.full_sampling:
            if self.n_patches != 1:
                print("Warning: number of patches is set to 1 for full sampling. Setting n_patches to 1.")
                self.n_patches = 1
        
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
    
    def __getitem__(self, idx):
        out = self.__getpatch__(idx)
        out['gt'] = out['gt'].unsqueeze(0)  # Add batch dimension
        out['gt_coord'] = out['gt_coord'].unsqueeze(0)  # Add batch dimension
        out['gt_cell'] = out['gt_cell'].unsqueeze(0)  # Add batch dimension
        
        for i in range(self.n_patches-1):
            temp = self.__getpatch__(idx)
            out['gt'] = torch.cat((out['gt'], temp['gt'][None]), dim=0)
            out['gt_coord'] = torch.cat((out['gt_coord'], temp['gt_coord'][None]), dim=0)
            out['gt_cell'] = torch.cat((out['gt_cell'], temp['gt_cell'][None]), dim=0)
            
        out['idx'] = idx
        return out
    
    def __getpatch__(self, idx):
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
            gt_res = random.randint(max(self.resize_gt_lb,orig_size//self.max_downscale), self.resize_gt_ub)
        
        # Resize ground truth
        if not self.full_sampling:
            resize_transform = self.resize(gt_res, antialias=True)
            gt = resize_transform(img.unsqueeze(0)).squeeze(0)
            gt = self._preprocess_tensor(gt)
        else:
            gt = self._preprocess_tensor(img)
        
        # Random crop patch from ground truth
        if self.full_sampling:
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
            
        return {
            'inp': inp,                    # C×H×W normalized
            'gt': gt,                      # C×P×P normalized
            'gt_coord': coord,             # P×P×2
            'gt_cell': cell,               # P×P×2
        }
        
    def visualize_item(self, samples=None, idx=None):
        if samples is None and idx is None:
            raise ValueError("Either samples or idx must be provided.")
        if samples is None:
            samples = self.__getitem__(idx)
            
        # number of patches
        n_patches = samples['gt'].shape[0]

        # Set up the figure
        fig = plt.figure(figsize=(20, 5 * n_patches))

        for i in range(n_patches):

            # 1. Original Input Image
            ax = plt.subplot(n_patches, 5, i * 5 + 1)
            img = self.tensors[samples['idx']]  # CHW format, uint8
            # Convert from [-1,1] to [0,1] for visualization
            plt.imshow(img.permute(1, 2, 0).numpy(), cmap='Grays')
            ax.set_title(f'Original Input ({img.shape[1]}x{img.shape[2]})')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)

            # 2. Encoder Input
            ax = plt.subplot(n_patches, 5, i * 5 + 2)
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
            ax = plt.subplot(n_patches, 5, i * 5 + 3)
            patch = samples['gt'][i]
            plt.imshow(patch.permute(1, 2, 0).numpy(), cmap='Grays')
            ax.set_title('Random GT Patch')
            ax.axis('off')

            # 4. Coordinate Grid Visualization
            ax = plt.subplot(n_patches, 5, i * 5 + 4)
            coord = (samples['gt_coord'][i].reshape(-1, 2) + 1 )/2
            sizes = samples['gt_cell'][i].reshape(-1, 2)/2
            patch_vals = samples['gt'][i].reshape(-1)
            plt.imshow(inp_viz.permute(1, 2, 0).numpy(), cmap='Grays')
            # plt.scatter(coord[:,0].numpy()*dataset.encoder_res, coord[:,1].numpy()*dataset.encoder_res, c='r', s=1)
            # draw square around each cell
            for j in range(len(coord)):
                x, y = coord[j] * self.encoder_res - sizes[j][0]* self.encoder_res/2
                s = sizes[j] * self.encoder_res
                face_alpha = f"#0000ff{int((patch_vals[j].numpy()+1)/2*255):02x}"
                rect = plt.Rectangle((x-s[0]/2, y-s[1]/2), s[0], s[1], linewidth=1, edgecolor='r', facecolor=face_alpha)
                ax.add_patch(rect)

            plt.tight_layout()
            