import warnings
warnings.filterwarnings("ignore")

from argparse import ArgumentParser
import os
import numpy as np
import torch
from OAT.DataUtils.DiffDataset import load_OAT_CDiff
from OAT.Trainers.Trainer import Trainer
from OAT.Diffusion.UNet import CTOPUNet
from OAT.Models.NFAE import NFAE
from OAT.Diffusion.diffusion import DDIMPipeline
from functools import partial
from OAT.InferenceUtils import INFD_Latent_Map
from tqdm.auto import tqdm, trange
import pickle
torch.set_float32_matmul_precision('high')

torch.autograd.set_grad_enabled(False)

args = ArgumentParser()
args.add_argument("--dataset", type=str, default="DOMTopoDiff", help="dataset name. Options: labeled, DOMTopoDiff")
args.add_argument("--dataset_path", type=str, default="Dataset", help="path to dataset. default Dataset")
args.add_argument("--n_samples", type=int, default=4, help="number of samples to generate per test case. default 4")
args.add_argument("--compile", action="store_true", help="compile the model")
args.add_argument("--mixed_precision", action="store_true", help="use mixed precision")
args.add_argument("--unet_path", type=str, default="CheckpointsDiff/last", help="path to UNet checkpoint. default CheckpointsDiff/last will find the latest checkpoint in Checkpoints")
args.add_argument("--ae_path", type=str, default="Checkpoints/last", help="path to AE checkpoint. default Checkpoints/last will find the latest checkpoint in Checkpoints")
args.add_argument("--guidance_scale", type=float, default=4.0, help="guidance scale. default 4.0")
args.add_argument("--save_path", type=str, default="Results", help="path to save results. default Results")
args.add_argument("--save_name", type=str, default="samples.pkl", help="name of the saved results. default samples.pkl")
args.add_argument("--start_idx", type=int, default=0, help="start index for inference. default 0")
args.add_argument("--end_idx", type=int, default=None, help="end index for inference. default None (end)")
args.add_argument("--num_sampling_steps", type=int, default=20, help="number of sampling steps. default 20")
args.add_argument('--ignore_BCs', action='store_true', help='Ignore BCs. Default: False')
args.add_argument('--ignore_vfs', action='store_true', help='Ignore vfs. Default: False')
args.add_argument("--seed", type=int, default=0, help="random seed. default 0")

args = args.parse_args()

torch.manual_seed(args.seed)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

dataset = load_OAT_CDiff(latents_path=None,
                         data_path=args.dataset_path,
                         subset=args.dataset,
                         split='test',
                         unconditional_prob=0,
                         BC_dropout_prob=0,
                         C_dropout_prob=0,
                         ignore_BC=args.ignore_BCs,
                         ignore_vf=args.ignore_vfs)

model = CTOPUNet.from_pretrained(args.unet_path).cuda()
model.eval()

AE = NFAE.from_pretrained(args.ae_path).cuda()
AE.eval()

diffusion = DDIMPipeline()

if args.compile:
    model.compile()
    AE.compile()
    
indices = np.arange(args.start_idx, len(dataset) if args.end_idx is None else args.end_idx)
    
batch_size = args.n_samples
results = []
if args.mixed_precision:
    with torch.amp.autocast('cuda'):
        for i in tqdm(indices):
            rnd_idx = i
            AE_batch = dataset.get_AE_grid(rnd_idx, batch_size)
            Diff_batch = dataset._collate_fn([dataset[rnd_idx]] * batch_size)
            Diff_batch = Diff_batch.to('cuda')
            AE_batch = AE_batch.to('cuda')
            
            latent_map = partial(INFD_Latent_Map, 
                            AE=AE, 
                            coords = AE_batch['gt_coord'], 
                            cells=AE_batch['gt_cell'], 
                            inverse_normalize=None)
            
            out = diffusion.inference(model, num_sampling_steps=args.num_sampling_steps, batch_size=batch_size, final_callable=latent_map, conditioning_over_relaxation_factor=args.guidance_scale, **Diff_batch)
            
            results.append(out)
else:
    for i in tqdm(indices):
        rnd_idx = i
        AE_batch = dataset.get_AE_grid(rnd_idx, batch_size)
        Diff_batch = dataset._collate_fn([dataset[rnd_idx]] * batch_size)
        Diff_batch.pop('sample');
        Diff_batch = Diff_batch.to('cuda')
        AE_batch = AE_batch.to('cuda')
        
        latent_map = partial(INFD_Latent_Map, 
                        AE=AE, 
                        coords = AE_batch['gt_coord'], 
                        cells=AE_batch['gt_cell'], 
                        inverse_normalize=None)
        
        out = diffusion.inference(model, num_sampling_steps=args.num_sampling_steps, batch_size=batch_size, final_callable=latent_map, conditioning_over_relaxation_factor=args.guidance_scale, **Diff_batch)
        
        results.append(out)
    
with open(os.path.join(args.save_path, args.save_name), 'wb') as f:
    pickle.dump(results, f)
print(f"Results saved to {os.path.join(args.save_path, args.save_name)}")
print("Inference completed.")