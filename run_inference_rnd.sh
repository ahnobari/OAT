#!/bin/bash

python run_inference_batch.py \
    --compile \
    --mixed_precision \
    --unet_path OpenTO/TODiffusion-Full \
    --ae_path OpenTO/TOAE-pretrained \
    --num_sampling_steps 5 \
    --guidance_scale 4.0 \
    --save_name RND_DDIM_5_4.pkl \
    --batch_size 64 \
    --save_image \
    --image_path RND_DDIM_5_4.png \
    --dataset labeled \
    --n_samples 4

python run_inference_batch.py \
    --compile \
    --mixed_precision \
    --unet_path OpenTO/TODiffusion-Full \
    --ae_path OpenTO/TOAE-pretrained \
    --num_sampling_steps 10 \
    --guidance_scale 4.0 \
    --save_name RND_DDIM_10_4.pkl \
    --batch_size 64 \
    --save_image \
    --image_path RND_DDIM_10_4.png \
    --dataset labeled \
    --n_samples 4

python run_inference_batch.py \
    --compile \
    --mixed_precision \
    --unet_path OpenTO/TODiffusion-Full \
    --ae_path OpenTO/TOAE-pretrained \
    --num_sampling_steps 20 \
    --guidance_scale 4.0 \
    --save_name RND_DDIM_20_4.pkl \
    --batch_size 64 \
    --save_image \
    --image_path RND_DDIM_20_4.png \
    --dataset labeled \
    --n_samples 4

python run_inference_batch.py \
    --compile \
    --mixed_precision \
    --unet_path OpenTO/TODiffusion-Full \
    --ae_path OpenTO/TOAE-pretrained \
    --num_sampling_steps 40 \
    --guidance_scale 4.0 \
    --save_name RND_DDIM_40_4.pkl \
    --batch_size 64 \
    --save_image \
    --image_path RND_DDIM_40_4.png \
    --dataset labeled \
    --n_samples 4

python run_inference_batch.py \
    --compile \
    --mixed_precision \
    --unet_path OpenTO/TODiffusion-Full \
    --ae_path OpenTO/TOAE-pretrained \
    --num_sampling_steps 5 \
    --guidance_scale 4.0 \
    --save_name RND_DDPM_5_4.pkl \
    --batch_size 64 \
    --save_image \
    --image_path RND_DDPM_5_4.png \
    --dataset labeled \
    --n_samples 4 \
    --ddpm

python run_inference_batch.py \
    --compile \
    --mixed_precision \
    --unet_path OpenTO/TODiffusion-Full \
    --ae_path OpenTO/TOAE-pretrained \
    --num_sampling_steps 10 \
    --guidance_scale 4.0 \
    --save_name RND_DDPM_10_4.pkl \
    --batch_size 64 \
    --save_image \
    --image_path RND_DDPM_10_4.png \
    --dataset labeled \
    --n_samples 4 \
    --ddpm

python run_inference_batch.py \
    --compile \
    --mixed_precision \
    --unet_path OpenTO/TODiffusion-Full \
    --ae_path OpenTO/TOAE-pretrained \
    --num_sampling_steps 20 \
    --guidance_scale 4.0 \
    --save_name RND_DDPM_20_4.pkl \
    --batch_size 64 \
    --save_image \
    --image_path RND_DDPM_20_4.png \
    --dataset labeled \
    --n_samples 4 \
    --ddpm

python run_inference_batch.py \
    --compile \
    --mixed_precision \
    --unet_path OpenTO/TODiffusion-Full \
    --ae_path OpenTO/TOAE-pretrained \
    --num_sampling_steps 40 \
    --guidance_scale 4.0 \
    --save_name RND_DDPM_40_4.pkl \
    --batch_size 64 \
    --save_image \
    --image_path RND_DDPM_40_4.png \
    --dataset labeled \
    --n_samples 4 \
    --ddpm