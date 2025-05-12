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
import matplotlib.pyplot as plt
import pickle
torch.set_float32_matmul_precision('high')

torch.autograd.set_grad_enabled(False)

args = ArgumentParser()
args.add_argument("--dataset", type=str, default="DOMTopoDiff", help="dataset name. Options: labeled, DOMTopoDiff")
args.add_argument("--dataset_path", type=str, default="Dataset", help="path to dataset. default Dataset")
args.add_argument("--n_samples", type=int, default=4, help="number of samples to generate per test case. default 4")
args.add_argument("--batch_size", type=int, default=8, help="batch size. default 8")
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
args.add_argument("--save_image", action="store_true", help="save a sample image of some generated samples")
args.add_argument("--image_path", type=str, default="inference_run.png", help="path to save image. default inference_run.png")
args.add_argument("--ddpm", action="store_true", help="use ddpm scheduler. default False")

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
B = args.batch_size
n_steps = (len(indices) + B - 1) // B
results = []
if args.mixed_precision:
    with torch.amp.autocast('cuda'):
        for i in trange(n_steps):
            start_idx = i * B
            end_idx = min((i + 1) * B, len(indices))
            
            # run decoder later
            latent_map = None
            
            batch = []
            for j in range(start_idx, end_idx):
                batch += [dataset[j]]*batch_size
            Diff_batch = dataset._collate_fn(batch)
            Diff_batch = Diff_batch.to('cuda')
            
            out = diffusion.inference(model, num_sampling_steps=args.num_sampling_steps, batch_size=len(batch), final_callable=latent_map, conditioning_over_relaxation_factor=args.guidance_scale, ddpm=args.ddpm, **Diff_batch)
            
            results.append(out.reshape(-1, batch_size, *out.shape[1:]))
            
        results = np.concatenate(results, axis=0)
        
        print("Running Decoder...")
        
        final_results = []
        for i in trange(results.shape[0]):
            AE_batch = dataset.get_AE_grid(indices[i], batch_size)
            inp = torch.tensor(results[i], device='cuda')
            AE_batch = AE_batch.to('cuda')
            out = INFD_Latent_Map(inp, AE, AE_batch['gt_coord'], AE_batch['gt_cell'], inverse_normalize=None).cpu().numpy()
            final_results.append(out)
else:
    for i in trange(n_steps):
        start_idx = i * B
        end_idx = min((i + 1) * B, len(indices))
        
        # run decoder later
        latent_map = None
        
        batch = []
        for j in range(start_idx, end_idx):
            batch += [dataset[j]]*batch_size
        Diff_batch = dataset._collate_fn(batch)
        Diff_batch = Diff_batch.to('cuda')
        
        out = diffusion.inference(model, num_sampling_steps=args.num_sampling_steps, batch_size=len(batch), final_callable=latent_map, conditioning_over_relaxation_factor=args.guidance_scale, ddpm=args.ddpm, **Diff_batch)
        
        results.append(out.reshape(-1, batch_size, *out.shape[1:]))
            
    results = np.concatenate(results, axis=0)
    final_results = []
    print("Running Decoder...")
    
    for i in trange(results.shape[0]):
        AE_batch = dataset.get_AE_grid(indices[i], batch_size)
        inp = torch.tensor(results[i], device='cuda')
        AE_batch = AE_batch.to('cuda')
        out = INFD_Latent_Map(inp, AE, AE_batch['gt_coord'], AE_batch['gt_cell'], inverse_normalize=None).cpu().numpy()
        final_results.append(out)
        

    
with open(os.path.join(args.save_path, args.save_name), 'wb') as f:
    pickle.dump(final_results, f)
print(f"Results saved to {os.path.join(args.save_path, args.save_name)}")
print("Inference completed.")

if args.save_image:
    n_rows = 10
    n_cols = batch_size + 1

    rnd_idx = np.random.randint(0, len(final_results), size=n_rows)

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols*8, n_rows*8))

    for i in range(n_rows):
        idx = rnd_idx[i]
        
        # Ground truth
        ax[i, 0].imshow(dataset.tensors[idx].numpy()[0], cmap='Grays')
        ax[i, 0].axis('off')
        ax[i, 0].set_title("Ground Truth")
        
        for j in range(batch_size):
            ax[i, j + 1].imshow(final_results[idx][j][0], cmap='Grays')
            ax[i, j + 1].axis('off')
            ax[i, j + 1].set_title(f"Sample {j+1}")
            
    plt.tight_layout()
    plt.savefig(args.image_path, dpi=300)