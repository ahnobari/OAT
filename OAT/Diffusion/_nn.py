
import torch
import torch.nn as nn
import torch.nn.functional as F

class BC_Encoder(nn.Module):
    '''
    Boundary-Condition Encoder (point cloud based)
    '''
    def __init__(self, mlp_layers):
        super(BC_Encoder, self).__init__()
        self.mlp = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for i in range(len(mlp_layers) - 1):
            self.mlp.append(nn.Linear(mlp_layers[i], mlp_layers[i+1]))
            # Add Layer Normalization except for the last layer
            if i < len(mlp_layers) - 2:
                self.layer_norms.append(nn.LayerNorm(mlp_layers[i+1]))

    def forward(self, positions, batch_index):
        # Apply MLP with Layer Normalization and ReLU to positions
        x = positions
        for i, layer in enumerate(self.mlp):
            x = layer(x)
            if i < len(self.mlp) - 1:  # Apply normalization and ReLU except for the last layer
                x = self.layer_norms[i](x)
                x = F.relu(x)

        batch_size = int(batch_index.max().item() + 1)
        pooled = []
        for index in range(batch_size):
            mask = (batch_index == index).squeeze()
            batch_x = x[mask]

            mean_pool = batch_x.mean(dim=0).squeeze()
            max_pool = batch_x.max(dim=0)[0].squeeze()
            min_pool = batch_x.min(dim=0)[0].squeeze()

            # Concatenate pooled features
            pooled_features = torch.cat((mean_pool, max_pool, min_pool), dim=0)
            pooled.append(pooled_features)

        # Stack pooled outputs for each set in the batch
        output = torch.stack(pooled)
    
        return output
    
class C_Encoder(nn.Module):
    '''
    Condition Encoder
    '''
    def __init__(self, mlp_layers):
        super(C_Encoder, self).__init__()
        self.mlp = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for i in range(len(mlp_layers) - 1):
            self.mlp.append(nn.Linear(mlp_layers[i], mlp_layers[i+1]))
            # Add Layer Normalization except for the last layer
            if i < len(mlp_layers) - 2:
                self.layer_norms.append(nn.LayerNorm(mlp_layers[i+1]))

    def forward(self, inputs):
        # Apply MLP with Layer Normalization and ReLU to inputs
        x = inputs
        for i, layer in enumerate(self.mlp):
            x = layer(x)
            if i < len(self.mlp) - 1:
                x = self.layer_norms[i](x)
                x = F.relu(x)
        return x