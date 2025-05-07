import warnings
warnings.filterwarnings("ignore")

from argparse import ArgumentParser
import os
import numpy as np
import torch
from OAT.DataUtils.DiffDataset import load_OAT_CDiff
from OAT.Trainers.Trainer import Trainer
from OAT.Diffusion.UNet import CTOPUNet
from OAT.Diffusion.diffusion import DDIMPipeline
torch.set_float32_matmul_precision('high')

args = ArgumentParser()
args.add_argument("--dataset", type=str, default="labeled", help="dataset name. Options: labeled, DOMTopoDiff")
args.add_argument("--dataset_path", type=str, default="Dataset", help="path to dataset. default Dataset")
args.add_argument("--latents_path", type=str, default=None, help="path to latents. Must be provided.")
args.add_argument("--num_epochs", type=int, default=50, help="number of epochs to train, default 50")
args.add_argument("--batch_size", type=int, default=32, help="batch size. default 32")
args.add_argument("--lr", type=float, default=1e-4, help="learning rate. default 1e-4")
args.add_argument("--cosine_scheduler", action="store_true", help="use cosine scheduler")
args.add_argument("--warmup_steps", type=int, default=200, help="number of warmup steps. default 200")
args.add_argument("--final_lr", type=float, default=1e-5, help="final learning rate. default 1e-5")
args.add_argument("--multi_gpu", action="store_true", help="use multiple GPUs")
args.add_argument("--DDP", action="store_true", help="use DDP")
args.add_argument("--compile", action="store_true", help="compile the model")
args.add_argument("--mixed_precision", action="store_true", help="use mixed precision")
args.add_argument("--checkpoint_dir", type=str, default="CheckpointsDiff", help="directory to save checkpoints. default CheckpointsDiff")
args.add_argument("--resume", action="store_true", help="resume training from checkpoint")
args.add_argument("--resume_path", type=str, default="CheckpointsDiff/last", help="path to resume checkpoint. default CheckpointsDiff/last will find the latest checkpoint in Checkpoints")
args.add_argument("--resume_model_only", action="store_true", help="resume training from checkpoint but do not load optimizer state and other states")
args.add_argument('--optimizer', type=str, default='AdamW', help='Optimizer to use. Default: AdamW, Options: Adam, AdamW, SGD, Adam8, Adafactor')
args.add_argument('--unconditional_prob', type=float, default=0.5, help='Unconditional probability. Default: 0.5')
args.add_argument('--BC_dropout_prob', type=float, default=0.25, help='BC dropout probability. Default: 0.25')
args.add_argument('--C_dropout_prob', type=float, default=0.25, help='C dropout probability. Default: 0.25')
args.add_argument('--ignore_BCs', action='store_true', help='Ignore BCs. Default: False')
args.add_argument('--ignore_vfs', action='store_true', help='Ignore vfs. Default: False')
args = args.parse_args()

if args.latents_path is None:
    raise ValueError("Latents path must be provided.")

dataset = load_OAT_CDiff(latents_path=args.latents_path,
                         data_path=args.dataset_path,
                         subset=args.dataset,
                         unconditional_prob=args.unconditional_prob, 
                         BC_dropout_prob=args.BC_dropout_prob, 
                         C_dropout_prob=args.C_dropout_prob,
                         ignore_BC=args.ignore_BCs,
                         ignore_vf=args.ignore_vfs)

model = CTOPUNet(
    sample_size       = dataset.latent_tensors.shape[2],          # latent spatial size
    in_channels       = dataset.latent_tensors.shape[1],           # VAE latent channels
    out_channels      = dataset.latent_tensors.shape[1],          # VAE latent channels
    layers_per_block  = 2,
    block_out_channels= (320, 640, 1280, 1280),  # matches SD‑v1
    down_block_types  = (
        "DownBlock2D",        # 64 → 32
        "DownBlock2D",        # 32 → 16
        "AttnDownBlock2D",    # 16 → 8  (self‑attention)
        "AttnDownBlock2D",    # 8  → 4
    ),
    up_block_types    = (
        "AttnUpBlock2D",      # 4  → 8
        "AttnUpBlock2D",      # 8  → 16
        "UpBlock2D",          # 16 → 32
        "UpBlock2D",          # 32 → 64
    ),
    norm_num_groups   = 32,
    mid_block_type    = "UNetMidBlock2DAttn",     # self‑attention in the bottleneck
    attention_head_dim= 8,                        # identical to SD‑v1
    BCs = [4,4] if not args.ignore_BCs else [],
    BC_n_layers = [4,4] if not args.ignore_BCs else [],
    BC_hidden_size = [256,256] if not args.ignore_BCs else [], 
    BC_emb_size = [128,128] if not args.ignore_BCs else [], 
    Cs = [1,2,1] if not args.ignore_vfs else [2,1],
    C_n_layers = [4,4,4] if not args.ignore_vfs else [4,4],
    C_hidden_size = [256,256,256] if not args.ignore_vfs else [256,256],
    C_mapping_size = [128,128,128] if not args.ignore_vfs else [128,128],
    latent_size = 1280,
    latentShift=dataset.shift,
    latentScale=dataset.scale
)

diffusion = DDIMPipeline()


max_scheduler_steps = (args.num_epochs * len(dataset) + args.batch_size - 1) // args.batch_size

if args.multi_gpu:
    n_devices = torch.cuda.device_count()
    max_scheduler_steps = (max_scheduler_steps + n_devices - 1) // n_devices

trainer = Trainer(model, 
                  multi_gpu=args.multi_gpu,
                  DDP_train=args.DDP,
                  lr=args.lr,
                  cosine_schedule=args.cosine_scheduler,
                  warmup_steps=args.warmup_steps,
                  lr_final=args.final_lr, 
                  schedule_max_steps=max_scheduler_steps,
                  Compile=args.compile,
                  mixed_precision=args.mixed_precision,
                  optimizer=args.optimizer,
                  custom_loss=diffusion.compute_loss)

if args.resume:
    if os.path.exists(args.resume_path):
        trainer.load_checkpoint(args.resume_path, model_only=args.resume_model_only)
    elif "/last" in args.resume_path:
        # find the latest checkpoint in the resume directory
        r_dir = args.resume_path.replace("/last", "/")

        if os.path.exists(r_dir):
            checkpoints = [f for f in os.listdir(r_dir) if f.endswith(".pth")]
            if len(checkpoints) > 0:
                dates_modified = [os.path.getmtime(os.path.join(r_dir, f)) for f in checkpoints]
                latest_checkpoint = checkpoints[np.argmax(dates_modified)]
                trainer.load_checkpoint(os.path.join(r_dir, latest_checkpoint), model_only=args.resume_model_only)
                print(f"Resuming from checkpoint: {latest_checkpoint}")
            else:
                print("No checkpoint found in the directory")

trainer.train(dataset, 
              data_idx=np.arange(len(dataset)), 
              batch_size=args.batch_size,
              epochs=args.num_epochs,
              checkpoint_dir=args.checkpoint_dir,
              continue_loop=True,
              verbose=True,
              dict_input=True)