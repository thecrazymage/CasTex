#!/bin/bash

source deactivate
conda activate castex

BLENDER_PATH="objaverse_eval/blender-3.3.21-linux-x64/blender"
PYTHON_SCRIPT_PATH="objaverse_eval/render_utils/blender_script.py"
PATH_TO_BACKGROUND_IMAGE="objaverse_eval/assets/background.png"
ENV_MAP_PATH="objaverse_eval/assets/studio_small_06_2k.hdr"
BASE_OUTPUT_ROOT="objaverse_eval/renders"

if (( $# < 2 )); then
    echo "Usage: bash render_frames.sh <input_folder> <stage> [custom_output_name]"
    exit 1
fi

INPUT_ROOT_DIR="${1%/}"
STAGE="$2"
if [ -n "${3:-}" ]; then
    RUN_NAME="$3"
else
    RUN_NAME=$(basename "$INPUT_ROOT_DIR")
fi
OUTPUT_DIR="$BASE_OUTPUT_ROOT/$RUN_NAME"

render_models() {
    local input_root="$1"
    local output_root="$2"
    local stage="$3"
    local trajectory="$4"
    
    for model_dir in "$input_root"/*/; do
        [ -d "$model_dir" ] || continue
        local stage_dir="$model_dir/model_stage_$stage"
        
        if [ ! -d "$stage_dir" ]; then
            echo "Error: The stage folder '$stage_dir' could not be found for the model '$(basename "$model_dir")'." >&2
            continue
        fi
        
        local obj_file=$(find "$stage_dir" -maxdepth 1 -type f -name "*.obj" | head -n 1)

        if [ -z "$obj_file" ]; then
            echo "Error: No .obj file found in $stage_dir. Skipping." >&2
            continue
        fi

        echo "Processing: $(basename "$model_dir")"

        "$BLENDER_PATH" -b -noaudio -P "$PYTHON_SCRIPT_PATH" -- \
            --object_path "$obj_file" \
            --output_dir "$output_root" \
            --engine "CYCLES" \
            --trajectory "$trajectory" \
            --camera_dist 1.4 \
            --env_map_path "$ENV_MAP_PATH" \
            --env_map_strength 0.7 \
            --device OPTIX
    done
}

compile_gif() {
    local folder="$1"
    local output_folder="$2"
    local model_name=$(basename $folder)
    
    echo "Generating video: $model_name"
    ffmpeg -framerate 24 -pattern_type glob -i "$folder*.png" -i $PATH_TO_BACKGROUND_IMAGE -filter_complex "[1:v][0:v]overlay" -pix_fmt yuv420p -b 1200k "$output_folder/$model_name".mp4
}

traverse_directories_and_compile_gifs() {
    local dir="$1"
    local gif_destination_dir="$2"
    
    mkdir -p "$gif_destination_dir"
    
    for subdir in "$dir"/*/; do
        if [[ -d "$subdir" ]]; then
            compile_gif "$subdir" "$gif_destination_dir"
        fi
    done
}

echo "Step 1: Rendering frames for FID (frames)..."
render_models "$INPUT_ROOT_DIR" "$OUTPUT_DIR" "$STAGE" "frames"
bash scripts/sanity_check.sh -d "$OUTPUT_DIR/frames" -ef 20 -ed 0 -esf 0

echo "Step 2: Rendering frames for video (video)..."
render_models "$INPUT_ROOT_DIR" "$OUTPUT_DIR" "$STAGE" "video"
bash scripts/sanity_check.sh -d "$OUTPUT_DIR/video" -ef 60 -ed 0 -esf 0

echo "Step 3: Compiling videos..."
traverse_directories_and_compile_gifs "${OUTPUT_DIR}/video" "${OUTPUT_DIR}/mp4"
if [ $(find "${OUTPUT_DIR}/mp4" -maxdepth 1 -name "*.mp4" | wc -l) -eq 410 ]; then
    echo -e "\n\n\033[1;32m✓ All videos successfully compiled! (410 files)\033[0m\n\n"
else
    echo -e "\n\n\033[1;31m✗ Video compilation issue: Number of files is not equal to 410\033[0m\n\n"
fi

echo "Done!"