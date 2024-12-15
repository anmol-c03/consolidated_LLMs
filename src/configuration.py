from dataclasses import dataclass

@dataclass
class ViT_Config:
    image_size: int = 224
    patch_size: int = 16
    embed_dim: int=768
    ff_dim: int=768*4 
    num_heads: int=12
    layers: int=12

