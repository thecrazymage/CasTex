import torch
from pathlib import Path
from transformers import T5EncoderModel
from diffusers import DiffusionPipeline

def get_prompt_filename(prompt, directional=True):
    """Generates a standardized filename for caching text embeddings based on the prompt content."""
    if directional:
        return f'directional_embeddings_{prompt.replace(" ", "_")}.pt'
    else:
        return f'embeddings_{prompt.replace(" ", "_")}.pt'

def get_view_direction(thetas, phis):
    """
        Classifies camera positions into different views:
        0 - front view: phi in [pi/4, 3pi/4], 
        1 - side view: phi in (-pi/4, pi/4) or (3pi/4, 5pi/4), 
        2 - backside view: phi in [5pi/4, 7pi/4], 
        3 - top view: theta in [pi/4, 3pi/4], 
        4 - bottom view: theta in [5pi/4, 7pi/4]
    """
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    phis = phis % (2 * torch.pi)
    thetas = thetas % (2 * torch.pi)

    front_mask = (phis >= torch.pi / 4) & (phis <= 3 * torch.pi / 4)
    backside_mask = (phis >= 5 * torch.pi / 4) & (phis <= 7 * torch.pi / 4)
    side_mask = ((phis > 7 * torch.pi / 4) | (phis < torch.pi / 4)) | ((phis > 3 * torch.pi / 4) & (phis < 5 * torch.pi / 4))
    top_mask = (thetas >= torch.pi / 4) & (thetas <= 3 * torch.pi / 4)
    bottom_mask = (thetas >= 5 * torch.pi / 4) & (thetas <= 7 * torch.pi / 4)

    res[front_mask] = 0
    res[side_mask] = 1
    res[backside_mask] = 2
    res[top_mask] = 3
    res[bottom_mask] = 4

    return res

def encode_prompt(prompt, directional=True, deepfloyd_model='DeepFloyd/IF-I-XL-v1.0', cache_dir='cached_prompts', device='cuda:0'):
    """
        Generates and caches text embeddings from a prompt using a T5 encoder, 
        optionally creating view-dependent variations.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)
    
    embedding_path = cache_path / get_prompt_filename(prompt, directional)
    if embedding_path.exists():
        return torch.load(embedding_path)

    text_encoder = T5EncoderModel.from_pretrained(
        deepfloyd_model,
        subfolder="text_encoder",
        load_in_8bit=True,
        variant="8bit",
    )
    pipe = DiffusionPipeline.from_pretrained(
        deepfloyd_model,
        text_encoder=text_encoder,
        unet=None,
    )

    if directional:
        directions = ['front view', 'side view', 'backside view', 'top view', 'bottom view']
        prompts = [f"{prompt}, {direction}" for direction in directions]
    else:
        prompts = [prompt]

    pipe = pipe.to(device)
    prompt_embeddings = pipe.encode_prompt(prompts)
    torch.save(prompt_embeddings, embedding_path)
    
    return prompt_embeddings