import torch
import torch.nn as nn
from ._nn import BC_Encoder, C_Encoder

class ProblemEncoder(nn.Module):
    def __init__(self,
                 BCs = [4,4],
                 BC_n_layers = [4,4],
                 BC_hidden_size = [256,256], 
                 BC_emb_size = [64,64], 
                 Cs = [1,2],
                 C_n_layers = [4,4],
                 C_hidden_size = [256,256],
                 C_mapping_size = [128,128],
                 latent_size = 256,
                 **ignore_kwargs):
        
        super(ProblemEncoder, self).__init__()
        
        # check if BC_n_layers and BC_hidden_size are lists of same length
        assert len(BC_n_layers) == len(BC_hidden_size)
        assert len(BC_n_layers) == len(BC_emb_size)
        assert len(BC_n_layers) == len(BCs)
        
        # check if C_mapping_size is a list of same length as Cs
        assert len(C_n_layers) == len(Cs)
        assert len(C_n_layers) == len(C_mapping_size)
        assert len(C_n_layers) == len(C_hidden_size)
        
        latent_conditional_dim = 0
        
        if len(BCs) > 0:
            self.hasBC = True
            self.BC_Networks = nn.ModuleList()
            for i in range(len(BCs)):
                self.BC_Networks.append(BC_Encoder([BCs[i]] + [BC_hidden_size[i]]* BC_n_layers[i] + [BC_emb_size[i]]))
                latent_conditional_dim += BC_emb_size[i]*3
        else:
            self.hasBC = False
        
        if len(Cs) > 0:
            self.hasC = True
            self.C_Networks = nn.ModuleList()
            for i in range(len(Cs)):
                self.C_Networks.append(C_Encoder([Cs[i]] + [C_hidden_size[i]]* C_n_layers[i] + [C_mapping_size[i]]))
                latent_conditional_dim += C_mapping_size[i]
        else:
            self.hasC = False
            
        self.latent_conditional_dim = latent_conditional_dim
        
        self.linear1 = nn.Linear(latent_conditional_dim, latent_size)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(latent_size, latent_size)
        
        if not self.hasC and not self.hasBC:
            raise ValueError("Error: ProblemEncoder must have at least one BC or C")
        
            
    def forward(self, BCs=None, BC_Batch=None, Cs=None, **kwargs):
        
        if self.hasC:
            bs = Cs[0].shape[0]
            device = Cs[0].device
        else:
            bs = int(BC_Batch[0].max().item() + 1)
            device = BC_Batch[0].device
        
        current_idx = 0
        full_cond = torch.zeros(bs, self.latent_conditional_dim, device=device)
        
        if self.hasBC:
            BC_emb = []
            for i in range(len(BCs)):
                BC_emb.append(self.BC_Networks[i](BCs[i],BC_Batch[i]))
            
            BC_emb = torch.cat(BC_emb,-1)
            full_cond[:,current_idx:current_idx+BC_emb.shape[1]] = BC_emb
            current_idx += BC_emb.shape[1]
        
        if self.hasC:
            C_emb = []
            for i in range(len(Cs)):
                C_emb.append(self.C_Networks[i](Cs[i]))
            
            C_emb = torch.cat(C_emb,-1)
            full_cond[:,current_idx:current_idx+C_emb.shape[1]] = C_emb
            current_idx += C_emb.shape[1]
            
        out = self.linear1(full_cond)
        out = self.act(out)
        out = self.linear2(out)
        
        return out