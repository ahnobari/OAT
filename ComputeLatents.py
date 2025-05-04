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
args.add_argument("--batch_size", type=int, default=200, help="batch size. default 256")
args.add_argument("--multi_gpu", action="store_true", help="use multiple GPUs")
args.add_argument("--DDP", action="store_true", help="use DDP")
args.add_argument("--compile", action="store_true", help="compile the model")
args.add_argument("--mixed_precision", action="store_true", help="use mixed precision")
args.add_argument("--checkpoint", type=str, default="Checkpoints/last", help="path to resume checkpoint. default Checkpoints/last will find the latest checkpoint in Checkpoints")
args.add_argument("--save_path", type=str, default="Latents", help="path to save latents. default Latents.")
args = args.parse_args()


dataset = load_OAT_AE(data_path=args.dataset_path,
                      encoder_res=args.encoder_res,
                      subset=args.dataset,
                      full_sampling=True)

model = NFAE(resolution=args.encoder_res)

trainer = Trainer(model, 
                  multi_gpu=args.multi_gpu,
                  DDP_train=args.DDP,
                  Compile=args.compile,
                  mixed_precision=args.mixed_precision)
if os.path.exists(args.checkpoint):
    model.load_from_trainer_checkpoint(args.checkpoint, strict=False)
    latest_checkpoint = args.checkpoint.split("/")[-1]
    
elif "/last" in args.checkpoint:
    # find the latest checkpoint in the resume directory
    r_dir = args.checkpoint.replace("/last", "/")

    if os.path.exists(r_dir):
        checkpoints = [f for f in os.listdir(r_dir) if f.endswith(".pth")]
        if len(checkpoints) > 0:
            dates_modified = [os.path.getmtime(os.path.join(r_dir, f)) for f in checkpoints]
            latest_checkpoint = checkpoints[np.argmax(dates_modified)]
            model.load_from_trainer_checkpoint(os.path.join(r_dir, latest_checkpoint), strict=False)
            print(f"Loading from checkpoint: {latest_checkpoint}")
        else:
            print("No checkpoint found in the directory")

# Create the save directory if it doesn't exist
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    
latents = trainer.compute_latents(
    dataset,
    data_idx=np.arange(len(dataset)), 
    batch_size=args.batch_size,
)

if latents is not None:
    # Save the latents to a file
    if args.DDP:
        save_name = latest_checkpoint.replace(".pth", f"_latents_ddp_{trainer.rank}_{args.dataset}.pth")
    else:
        save_name = latest_checkpoint.replace(".pth", f"_latents_{args.dataset}.pth")
    torch.save(latents, os.path.join(args.save_path, save_name))
    print(f"Latents saved to {os.path.join(args.save_path, save_name)}")

if args.DDP:
    torch.distributed.barrier()
# wait for all processes to finish
if args.DDP and trainer.is_main_process():
    # combine latents from all processes
    all_latents = []
    for i in range(trainer.world_size):
        latents = torch.load(os.path.join(args.save_path, latest_checkpoint.replace(".pth", f"_latents_ddp_{i}_{args.dataset}.pth")))
        all_latents.append(latents)
    all_latents = torch.cat(all_latents, dim=0)
    # save combined latents
    torch.save(all_latents, os.path.join(args.save_path, latest_checkpoint.replace(".pth", f"_latents_{args.dataset}.pth")))
    print(f"Latents saved to {os.path.join(args.save_path, latest_checkpoint.replace('.pth', f'_latents_{args.dataset}.pth'))}")
    
    # delete individual latents
    for i in range(trainer.world_size):
        os.remove(os.path.join(args.save_path, latest_checkpoint.replace(".pth", f"_latents_ddp_{i}_{args.dataset}.pth")))
        print(f"Deleted {os.path.join(args.save_path, latest_checkpoint.replace('.pth', f'_latents_ddp_{i}_{args.dataset}.pth'))}")