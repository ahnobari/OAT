from diffusers import DDIMScheduler, DDPMScheduler
import torch
from typing import Optional, Callable
from .UNet import CTOPUNet
from tqdm.auto import tqdm, trange

class DDIMPipeline:
    def __init__(self, num_training_steps = 1000, prediction_type = 'v_prediction', rescale_betas_zero_snr=True, timestep_spacing='trailing', cosine_schedule=True, loss_type='l2'):
        
        self.TrainScheduler = DDPMScheduler(
            num_train_timesteps=num_training_steps,
            prediction_type=prediction_type,
            rescale_betas_zero_snr=rescale_betas_zero_snr,
            timestep_spacing=timestep_spacing,
            beta_schedule = 'linear' if not cosine_schedule else 'squaredcos_cap_v2'
        )
        
        self.InferenceScheduler = DDIMScheduler(
            num_train_timesteps=num_training_steps,
            beta_start=self.TrainScheduler.config.beta_start,
            beta_end=self.TrainScheduler.config.beta_end,
            beta_schedule=self.TrainScheduler.config.beta_schedule,
            prediction_type=prediction_type,
            rescale_betas_zero_snr=rescale_betas_zero_snr,
            timestep_spacing=timestep_spacing
        )
        
        self.prediction_type = prediction_type
        
        self.loss_fn = torch.nn.MSELoss() if loss_type == 'l2' else torch.nn.L1Loss()
        
    def get_target(self, x, noise, t):
        """
        Get the target for the model based on the prediction type.
        """
        if self.prediction_type == 'epsilon':
            return noise
        elif self.prediction_type == 'v_prediction':
            return self.TrainScheduler.get_velocity(x, noise, t)
        elif self.prediction_type == 'sample':
            return x
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
    
    def compute_loss(self, model: CTOPUNet, sample: torch.Tensor, noise: Optional[torch.Tensor] = None, **kwargs):
        
        bs = sample.shape[0]
        timesteps = torch.randint(
                    0, self.TrainScheduler.config.num_train_timesteps, (bs,), device=model.device,
                    dtype=torch.long
                )
        noise = torch.randn_like(sample) if noise is None else noise
        
        noisy_sample = self.TrainScheduler.add_noise(sample, noise, timesteps)
        
        pred = model(noisy_sample, timesteps, **kwargs)[0]
        
        target = self.get_target(sample, noise, timesteps)
        
        loss = self.loss_fn(pred, target)
        
        return {'loss': loss }
    
    @torch.no_grad()
    def inference(self, 
                  model: CTOPUNet,
                  num_sampling_steps: int = 50,
                  noise: Optional[torch.Tensor] = None,
                  batch_size: Optional[int] = 1,
                  guidance_function: Optional[Callable] = None,
                  guidance_scale: float = 1.0,
                  static_guidance: bool = False,
                  direct_guidance: bool = False,
                  final_callable: Optional[Callable] = None,
                  **kwargs):
        
        """
        Perform inference using the DDIM scheduler.
        """
        if noise is None:
            noise = torch.randn((batch_size, model.conv_in.in_channels, model.sample_size, model.sample_size), device=model.device)
        
        batch_size = noise.shape[0]
        
        # Prepare the scheduler
        self.InferenceScheduler.set_timesteps(num_sampling_steps)
        
        for t in tqdm(self.InferenceScheduler.timesteps):
            pred_noise = model(noise, t, **kwargs).sample
                
            denoised = self.InferenceScheduler.step(pred_noise, t, noise)
            pred = (denoised.pred_original_sample + 1) / 2 * model.latent_scale - model.latent_shift
            
            if guidance_function is not None:
                grads = guidance_function(pred)
            
                alpha_bar = self.InferenceScheduler.alphas_cumprod[t]        # scalar tensor
                beta_bar  = 1.0 - alpha_bar
                
                if static_guidance:
                    mult = guidance_scale
                else:
                    mult = guidance_scale * (beta_bar / alpha_bar).sqrt()
                
                if direct_guidance:
                    noise = denoised.prev_sample
                    noise = noise - mult * grads
                    
                else:
                    grad_eps  = -grads * (beta_bar / alpha_bar).sqrt()   # dε = −√β/α ∇x0
                    noise_pred = pred_noise + guidance_scale * grad_eps
                    noise = self.InferenceScheduler.step(noise_pred, t, noise).prev_sample
            else:
                noise = denoised.prev_sample
        
        noise = (noise + 1) / 2 * model.latent_scale - model.latent_shift
        
        if final_callable is not None:
            noise = final_callable(noise)
        
        noise = noise.detach().cpu().numpy()
        
        return noise