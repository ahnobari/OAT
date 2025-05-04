from diffusers import DDIMScheduler, DDPMScheduler
import torch
from typing import Optional
from .UNet import CTOPUNet

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