from ..Diffusion._models import ProblemEncoder
import torch
from torch import nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from .vision import build_clip_vision_L14

class LatentPhysx(nn.Module, PyTorchModelHubMixin):
    def __init__(self, 
                 image_size: int = 64,
                 patch_size: int = 8,
                 num_channels: int = 1,
                 projection_dim: int = 1280,
                 BCs = [4,4],
                 BC_n_layers = [4,4],
                 BC_hidden_size = [256,256], 
                 BC_emb_size = [64,64], 
                 Cs = [1,2],
                 C_n_layers = [4,4],
                 C_hidden_size = [256,256],
                 C_mapping_size = [128,128],
                 latentShift = 0,
                 latentScale = 1
                ):
        super().__init__()
        
        self.vision_model = build_clip_vision_L14(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            projection_dim=projection_dim,
        )
        self.visual_projection = nn.Linear(1024, projection_dim, bias=False)
        
        self.problem_encoder = ProblemEncoder(
            BCs = BCs,
            BC_n_layers = BC_n_layers,
            BC_hidden_size = BC_hidden_size, 
            BC_emb_size = BC_emb_size, 
            Cs = Cs,
            C_n_layers = C_n_layers,
            C_hidden_size = C_hidden_size,
            C_mapping_size = C_mapping_size,
            latent_size = projection_dim
        )
        
        self.tau = CLIPContrastiveLoss()
        
        self.latent_shift = torch.nn.Parameter(torch.tensor([latentShift]), requires_grad=False)
        self.latent_scale = torch.nn.Parameter(torch.tensor([latentScale]), requires_grad=False)
    
    def forward(self, sample, compute_loss = False, **kwargs):
        vision_latent = self.vision_model(pixel_values=sample).pooler_output
        vision_latent = self.visual_projection(vision_latent)
        
        problem_latent = self.problem_encoder(**kwargs)
        
        
        out = {
                'vision_latent': vision_latent,
                'problem_latent': problem_latent
              }
        
        if compute_loss:
            # compute loss
            loss = self.tau(vision_latent, problem_latent)
            loss_dict = {'loss': loss}
            
            return out, loss_dict
        
        return out

class CLIPContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # logit_scale is a single learnable scalar (initialized near 1/0.07)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1/0.07)))

    def forward(self, image_embeddings, text_embeddings):
        """
        image_embeddings: (N, D) ℓ₂‑normalized
        text_embeddings:  (N, D) ℓ₂‑normalized
        """
        # 1) scale
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_embeddings @ text_embeddings.t()  # (N, N) similarity matrix

        # 2) labels = [0,1,2,...,N-1]
        N = logits.shape[0]
        targets = torch.arange(N, device=logits.device)

        # 3) compute cross-entropy in both directions
        loss_i2t = F.cross_entropy(logits, targets)      # image→text
        loss_t2i = F.cross_entropy(logits.t(), targets)  # text→image

        # 4) symmetric average
        return (loss_i2t + loss_t2i) / 2