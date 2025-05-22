# Optimize Any Topology
This repository is the official implemantation of the OAT model for minimum compliance topology optimization.

***ADD IMAGE***

## Environment
To run this code create a new python environment and run:

```bash
pip install -r requirements.txt
```

> **NOTE**: To run the optimizer and FEA as efficiently as possible on Intel hardware please create an environment with MKL complied packages. To do this run:
```bash
bash mkl_setup.sh
```

## The Open-TO Dataset
The dataset of fully randomized optimized topologies we developed can be downloaded from HuggingFace. To download the data:

```bash
cd Dataset
python run_this.py
```

This dataset includes 2.2M total samples, with 900K labeled samples as well as 10K test data.

## Training The Autoencoder
To train the autoencoder run `trainAE.py`. To see options run help. One can run this with torchrun and pass `--DDP` and `--multi_gpu` arguments to train on multiple GPUs.

### Pre-Trained Checkpoints
Our checkpoint is availble on HF. This checkpoint is available at `Open-TO/AE`

## Training Diffusion Model
To train the latent diffusion first run `ComputeLatents.py` then run `trainDiff.py`. To see options run help. One can run this with torchrun and pass `--DDP` and `--multi_gpu` arguments to train on multiple GPUs.

### Pre-Trained Checkpoints
Our checkpoint is availble on HF. This checkpoint is available at `Open-TO/Diff`

# Inference For Test
To run the model we have trained to generate samples for the Open-TO test set run:

```bash
python run_inference_batch.py \
    --compile \
    --mixed_precision \
    --unet_path Open-TO/Diff \
    --ae_path Open-TO/AE \
    --num_sampling_steps 20 \
    --guidance_scale 2.0 \
    --save_name RND_DDIM_20_4.pkl \
    --batch_size 64 \
    --save_image \
    --image_path RND_DDIM_20_4.png \
    --dataset labeled \
    --n_samples 4
```

Once done you can run:

```bash
python run_compliance_test.py --jobs 48 --samples_path Results/RND_DDIM_20_4.pkl --save_name "RND_DDIM_20_4.pkl" --post_opt --dataset labeled
```

This code will output CE and VFE.

To the model on the full 5 shape benchmark of prior works run:

```bash
python run_inference_batch.py \
    --compile \
    --mixed_precision \
    --unet_path Open-TO/Diff \
    --ae_path Open-TO/AE \
    --num_sampling_steps 20 \
    --guidance_scale 2.0 \
    --save_name NITO_DDIM_20_4.pkl \
    --batch_size 64 \
    --save_image \
    --image_path NITO_DDIM_20_4.png \
    --dataset DOMTopoDiff \
    --n_samples 4
```

Then run:

```bash
python run_compliance_test.py --jobs 48 --samples_path "Results/NITO_DDIM_20_4.pkl" --save_name "NITO_DDIM_20_4.pkl" --post_opt
```