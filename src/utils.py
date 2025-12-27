import gc
import os
import cv2
import torch
import random
import numpy as np
from torchvision import transforms
from torch.nn import Parameter, ParameterDict

def flush():
    """Forces garbage collection and clears the CUDA memory cache."""
    gc.collect()
    torch.cuda.empty_cache()

def seed_all(seed):
    """Sets random seeds for Python, NumPy, and PyTorch to ensure reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def init_texture(texture_dim, diffuse_only=False, device='cuda:0', diffuse_path=None, no_diffuse_learning=False):
    """
        Creates and initializes the learnable texture parameters (diffuse, normal, metallic, roughness), 
        optionally loading an initial image.
    """
    if diffuse_path:
        diffuse_image = cv2.resize(
            cv2.cvtColor(cv2.imread(diffuse_path), cv2.COLOR_RGB2BGR), 
            (texture_dim, texture_dim)
        )
        diffuse_texture = Parameter(transforms.ToTensor()(diffuse_image).permute(1, 2, 0).contiguous()).to(device)
    else:
        diffuse_texture = torch.full((texture_dim, texture_dim, 3), 0.5)

    if not diffuse_only:
        normals_texture = torch.zeros((texture_dim, texture_dim, 3))
        normals_texture[..., 2] = 1.0
        metallic_texture = torch.full((texture_dim, texture_dim, 1), 0.5)
        roughness_texture = torch.full((texture_dim, texture_dim, 1), 0.5)

        material_parameters = ParameterDict({
            'diffuse_texture':   Parameter(diffuse_texture.to(device)),
            'normals_texture':   Parameter(normals_texture.to(device)),
            'metallic_texture':  Parameter(metallic_texture.to(device)),
            'roughness_texture': Parameter(roughness_texture.to(device)),
        })
    else:
        material_parameters = ParameterDict({
            'diffuse_texture':   Parameter(diffuse_texture.to(device)),
        })
    
    if no_diffuse_learning:
        material_parameters['diffuse_texture'].requires_grad_(False)

    return material_parameters

def save_texture(experiment_path, texture, stage):
    """Saves the current state of the texture tensors to disk."""
    save_dir = os.path.join(experiment_path, f'model_stage_{stage}')
    os.makedirs(save_dir, exist_ok=True)
    torch.save(texture, os.path.join(save_dir, f'texture_stage_{stage}.pt'))
    print(f"Stage {stage} textures saved to {save_dir}")