import warnings
warnings.filterwarnings("ignore")

from argparse import ArgumentParser
import os
import numpy as np
import torch
from OAT.DataUtils.AEDataset import load_OAT_AE
from matplotlib import pyplot as plt
from OAT.Trainers.Trainer import Trainer
from OAT.Models.NFAE import NFAE
torch.set_float32_matmul_precision('high')

args = ArgumentParser()
args.add_argument("--dataset", type=str, default="labeled", help="dataset name. Options: labeled, pre_training")
args.add_argument("--dataset_path", type=str, default="Dataset", help="path to dataset. default Dataset")
args.add_argument("--encoder_res", type=int, default=256, help="encoder resolution. default 256")
args.add_argument("--patch_size", type=int, default=64, help="patch size. default 64")
args.add_argument("--num_epochs", type=int, default=40, help="number of epochs to train, default 40")
args.add_argument("--batch_size", type=int, default=32, help="batch size. default 32")
args.add_argument("--lr", type=float, default=1e-4, help="learning rate. default 1e-4")
args.add_argument("--cosine_scheduler", action="store_true", help="use cosine scheduler")
args.add_argument("--final_lr", type=float, default=1e-5, help="final learning rate. default 1e-5")
args.add_argument("--multi_gpu", action="store_true", help="use multiple GPUs")
args.add_argument("--DDP", action="store_true", help="use DDP")
args.add_argument("--compile", action="store_true", help="compile the model")
args.add_argument("--mixed_precision", action="store_true", help="use mixed precision")
args.add_argument("--checkpoint_dir", type=str, default="Checkpoints", help="directory to save checkpoints")
args.add_argument("--resume", action="store_true", help="resume training from checkpoint")
args.add_argument("--resume_path", type=str, default="Checkpoints/last", help="path to resume checkpoint. default Checkpoints/last will find the latest checkpoint in Checkpoints")
args.add_argument("--resume_model_only", action="store_true", help="resume training from checkpoint but do not load optimizer state and other states")
args.add_argument('--optimizer', type=str, default='AdamW', help='Optimizer to use. Default: AdamW, Options: Adam, AdamW, SGD, Adam8, Adafactor')
args.add_argument('--max_sampling_prob', type=float, default=1.0, help='Maximum sampling probability (sample true resolution patch). Default: 1.0')
args.add_argument('--full_sampling', action='store_true', help='Use full sampling (sample entire image). Default: False. If True patch size will be image size')
args.add_argument("--hf_checkpoint", action="store_true", help="huggingface checkpoint is passed.")
args = args.parse_args()


dataset = load_OAT_AE(data_path=args.dataset_path,
                      encoder_res=args.encoder_res,
                      patch_size=args.patch_size,
                      resize_gt_lb=args.patch_size,
                      resize_gt_ub=2048,
                      p_max=args.max_sampling_prob,
                      subset=args.dataset,
                      full_sampling=args.full_sampling)

model = NFAE(resolution=args.encoder_res)

max_scheduler_steps = (args.num_epochs * len(dataset) + args.batch_size - 1) // args.batch_size

if args.multi_gpu:
    n_devices = torch.cuda.device_count()
    max_scheduler_steps = (max_scheduler_steps + n_devices - 1) // n_devices

trainer = Trainer(model, 
                  multi_gpu=args.multi_gpu,
                  DDP_train=args.DDP,
                  lr=args.lr,
                  cosine_schedule=args.cosine_scheduler,
                  lr_final=args.final_lr, 
                  schedule_max_steps=max_scheduler_steps,
                  Compile=args.compile,
                  mixed_precision=args.mixed_precision,
                  optimizer=args.optimizer)


if args.resume:
    if args.hf_checkpoint:
        model.from_pretrained(args.resume_path)
    elif os.path.exists(args.resume_path):
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
              verbose=True)