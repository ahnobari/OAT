from argparse import ArgumentParser
import os
import numpy as np
import torch
from OAT.DataUtils._AE_utils import load_OAT_AE
from matplotlib import pyplot as plt
from OAT.Trainers.Trainer import Trainer
from OAT.Models._models import ResFreeAutoEncoder

args = ArgumentParser()
args.add_argument("--dataset", type=str, default="labeled", help="dataset name. Options: labeled, pre_training")
args.add_argument("--encoder_res", type=int, default=96, help="encoder resolution. default 96")
args.add_argument("--patch_size", type=int, default=32, help="patch size. default 32")
args.add_argument("--num_epochs", type=int, default=50, help="number of epochs to train, default 50")
args.add_argument("--n_patches", type=int, default=1, help="number of patches. default 1")
args.add_argument("--batch_size", type=int, default=32, help="batch size. default 32")
args.add_argument("--lr", type=float, default=1e-4, help="learning rate. default 1e-4")
args.add_argument("--cosine_scheduler", action="store_true", help="use cosine scheduler")
args.add_argument("--final_lr", type=float, default=1e-5, help="final learning rate. default 1e-5")
args.add_argument("--multi_gpu", action="store_true", help="use multiple GPUs")
args.add_argument("--DDP", action="store_true", help="use DDP")
args.add_argument("--compile", action="store_true", help="compile the model")
args.add_argument("--checkpoint_dir", type=str, default="Checkpoints", help="directory to save checkpoints")
args.add_argument("--resume", action="store_true", help="resume training from checkpoint")
args.add_argument("--resume_path", type=str, default="Checkpoints/last", help="path to resume checkpoint. default Checkpoints/last will find the latest checkpoint in Checkpoints")

args = args.parse_args()


dataset = load_OAT_AE(encoder_res=args.encoder_res,
                        patch_size=args.patch_size,
                        resize_gt_lb=args.patch_size,
                        resize_gt_ub=1024,
                        n_patches=args.n_patches,
                        subset=args.dataset)
model = ResFreeAutoEncoder(ae_resolution=args.encoder_res)
trainer = Trainer(model, 
                  multi_gpu=args.multi_gpu,
                  DDP_train=args.DDP,
                  lr=args.lr,
                  cosine_schedule=args.cosine_scheduler,
                  lr_final=args.final_lr, 
                  schedule_max_steps=args.num_epochs,
                  Compile=args.compile,
                  mixed_precision=True)


if args.resume:
    if os.path.exists(args.resume_path):
        trainer.load_checkpoint(args.resume_path)
    elif "/last" in args.resume_path:
        # find the latest checkpoint in the resume directory
        r_dir = args.resume_path.replace("/last", "/")

        if os.path.exists(r_dir):
            checkpoints = [f for f in os.listdir(r_dir) if f.endswith(".pth")]
            if len(checkpoints) > 0:
                dates_modified = [os.path.getmtime(os.path.join(r_dir, f)) for f in checkpoints]
                latest_checkpoint = checkpoints[np.argmax(dates_modified)]
                trainer.load_checkpoint(os.path.join(r_dir, latest_checkpoint))
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