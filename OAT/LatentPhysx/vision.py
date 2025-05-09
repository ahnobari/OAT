from transformers import CLIPVisionConfig, CLIPVisionModel

def build_clip_vision_L14(
    image_size: int = 224,
    patch_size: int = 14,
    num_channels: int = 3,
    projection_dim: int = 768,
) -> CLIPVisionModel:
    config = CLIPVisionConfig(
        image_size=image_size,
        patch_size=patch_size,
        num_channels=num_channels,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        projection_dim=projection_dim,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
    )
    model = CLIPVisionModel(config)
    
    return model