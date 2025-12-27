import os
import gc
import cv2
import time
import yaml
import torch
import random
import argparse
from datetime import datetime

from src.train import Trainer
from src.mesh import load_mesh
from src.renderer import Renderer
from src.prompt_processing import encode_prompt
from src.utils import init_texture, seed_all, flush, save_texture

def parse_args():
    parser = argparse.ArgumentParser()
    
    # ========== Experiment Configuration ==========
    exp_group = parser.add_argument_group('Experiment Configuration')
    exp_group.add_argument('--description',     type=str, default='',       help='Experiment description for logging')
    exp_group.add_argument('--seed',            type=int, default=42,       help='Random seed for reproducibility')
    exp_group.add_argument('--log_dir',         type=str, default='logs',   help='Root directory for saving experiments')
    exp_group.add_argument('--device',          type=str, default='cuda:0', help='Device to run computations')
    exp_group.add_argument('--objaverse_eval',  action='store_true',        help='Use Objaverse evaluation mode')
    
    # ========== Input Data ==========
    data_group = parser.add_argument_group('Input Data')
    data_group.add_argument('--prompt',             type=str, required=True,            help='Text prompt describing desired texture')
    data_group.add_argument('--mesh_location',      type=str, default='data/spot.obj',  help='Path to input mesh file (.obj format)')
    data_group.add_argument('--use_dir_embeddings', action='store_true',                help='Use view-dependent text embeddings')
    data_group.add_argument('--prompt_cache_dir',   type=str, default='cached_prompts', help='Directory for caching encoded prompts')
    
    # ========== Diffusion Model ==========
    model_group = parser.add_argument_group('Diffusion Model')
    model_group.add_argument('--model_i',  type=str, default="DeepFloyd/IF-I-XL-v1.0", help='HuggingFace model ID for stage I')
    model_group.add_argument('--model_ii', type=str, default="DeepFloyd/IF-II-L-v1.0", help='HuggingFace model ID for stage II')
    
    # ========== Training Configuration ==========
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--num_steps_i',        type=int,   default=500,  help='Number of optimization steps for stage I')
    train_group.add_argument('--num_steps_ii',       type=int,   default=1000, help='Number of optimization steps for stage II')
    train_group.add_argument('--batch_size',         type=int,   default=8,    help='Number of views to render per training iteration')
    train_group.add_argument('--guidance_scale_i',   type=float, default=15.0, help='Classifier-free guidance scale for stage I')
    train_group.add_argument('--guidance_scale_ii',  type=float, default=10.0, help='Classifier-free guidance scale for stage II')
    train_group.add_argument('--lowres_noise_level', type=float, default=0.5,  help='Noise level for stage II upsampling (0.0-1.0)')
    train_group.add_argument('--eval_plot_iter',     type=int,   default=100,  help='Save visualization every N iterations')
    
    # ========== Rendering & Material ==========
    render_group = parser.add_argument_group('Rendering & Material')
    render_group.add_argument('--texture_dim',          type=int,   default=1024,                                         help='Texture resolution')
    render_group.add_argument('--env_light',            type=str,   default='assets/kloofendal_28d_misty_puresky_2k.hdr', help='Path to HDR environment map for lighting')
    render_group.add_argument('--light_scale',          type=float, default=2.0,                                          help='Multiplier for environment light intensity')
    render_group.add_argument('--diffuse_texture_path', type=str,   default=None,                                         help='Ready diffuse map for initialization')
    render_group.add_argument('--no_diffuse_learning',  action='store_true',                                              help='Train diffuse map or not')
    render_group.add_argument('--diffuse_only',         action='store_true',                                              help='Train only diffuse map')

    # ========== Optimizer ==========
    optim_group = parser.add_argument_group('Optimizer Adam')
    optim_group.add_argument('--lr_i',            type=float, default=0.01, help='Learning rate for stage I')
    optim_group.add_argument('--lr_ii',           type=float, default=0.01, help='Learning rate for stage II')
    optim_group.add_argument('--gamma_scheduler', type=float, default=0.9,  help='Multiplicative factor for exponential LR decay')
    
    args = parser.parse_args()
    return args

def main():

    start_time = time.time()

    args = parse_args()
    seed_all(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)
    
    prompt = args.prompt
    timestamp = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    experiment_name = f"{args.prompt}_{timestamp}"

    if args.objaverse_eval:
        parts = args.prompt.split("_")
        name = " ".join(parts[:-1])
        number = parts[-1]
        experiment_name = f"{name.replace(' ', '_')}_{number}"
        prompt = name
        args.prompt = name

    experiment_path = os.path.join(args.log_dir, experiment_name)
    args.experiment_path = experiment_path
    os.makedirs(experiment_path, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Experiment: {experiment_name}")
    print(f"Prompt: '{prompt}'")
    print(f"Output directory: {experiment_path}")
    print(f"{'='*80}\n")
    
    print("Encoding prompt embeddings...")
    prompt_embeddings = encode_prompt(
        prompt, 
        directional=args.use_dir_embeddings,
        cache_dir=args.prompt_cache_dir,
        device=args.device
    )
    
    print("Initializing textures...")
    texture = init_texture(
        texture_dim=args.texture_dim, 
        diffuse_only=args.diffuse_only, 
        device=args.device,
        diffuse_path=args.diffuse_texture_path,
        no_diffuse_learning=args.no_diffuse_learning
    )
    
    print("Loading mesh...")
    mesh = load_mesh(args.mesh_location, texture, use_predefined_texture=True if args.diffuse_texture_path else False)

    print("Loading renderer...")
    renderer = Renderer()
    
    phase1_time = 0.0
    if args.num_steps_i > 0:
        print(f"\n{'='*80}")
        print("Starting Stage i ...")
        print(f"{'='*80}")
        
        trainer = Trainer(mesh, texture, prompt_embeddings, renderer, 'i', args)
        start_stage1 = time.time()
        trainer.run_training_loop()
        
        if not args.objaverse_eval:
            save_texture(experiment_path, texture, 'i')
        
        del trainer
        flush()
        phase1_time = time.time() - start_stage1
    
    phase2_time = 0.0
    if args.num_steps_ii > 0:
        print(f"\n{'='*80}")
        print("Starting Stage ii ...")
        print(f"{'='*80}")
        
        trainer = Trainer(mesh, texture, prompt_embeddings, renderer, 'ii', args)
        start_stage2 = time.time()
        trainer.run_training_loop()
        
        if not args.objaverse_eval:
            save_texture(experiment_path, texture, 'ii')
        
        del trainer
        flush()
        phase2_time = time.time() - start_stage2
    
    total_time = time.time() - start_time
    args_dict = {
        **vars(args),
        'timestamp': timestamp,
        'phase_1_time': phase1_time,
        'phase_2_time': phase2_time,
        'total_time': total_time,
    }
    
    config_path = os.path.join(experiment_path, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(args_dict, f, default_flow_style=False)
    print(f"Configuration saved to {config_path}")
    
    print(f"\n{'='*80}")
    print("Experiment completed successfully!")
    print(f"Results saved to: {experiment_path}")
    print(f"Stage I time: {phase1_time:.2f}s")
    print(f"Stage II time: {phase2_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()