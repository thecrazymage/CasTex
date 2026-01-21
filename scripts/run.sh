#!/bin/bash

module load Python CUDA/12.4 gnu10
source deactivate
conda activate castex

PROMPT="A cow in a suit"
# PROMPT="A cow wearing a chef outfit"
# PROMPT="A cow in a Wonder Woman suit"
# PROMPT="A cow dressed as a doctor in a white coat"

python main.py \
    --mesh_location "assets/spot.obj" \
    --prompt "$PROMPT" \
    --num_steps_i 500 \
    --num_steps_ii 1000 \
    --guidance_scale_i 25 \
    --guidance_scale_ii 20 \
    --use_dir_embeddings