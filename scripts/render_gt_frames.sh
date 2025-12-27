#!/bin/bash

BLENDER_PATH="objaverse_eval/blender-3.3.21-linux-x64/blender"
PYTHON_SCRIPT_PATH="objaverse_eval/render_utils/blender_script.py"
DATA_PATH="objaverse_eval/objaverse_data/glbs"
OUTPUT_DIR="objaverse_eval/renders/ground_truth/"
ENV_MAP_PATH="objaverse_eval/assets/studio_small_06_2k.hdr"

# Function to process meshes
process_mesh() {
    local MESH_FILE="$1"
    echo "Processing mesh file: $MESH_FILE"
    "$BLENDER_PATH" -b -noaudio -P "$PYTHON_SCRIPT_PATH" -- \
        --object_path "$MESH_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --engine "CYCLES" \
        --trajectory "frames" \
        --camera_dist 1.4 \
        --env_map_path "$ENV_MAP_PATH" \
        --env_map_strength 0.7 \
        --device CPU
}
# We use CPU above because of problems with .glb and GPU rendering

if [ ! -d "$DATA_PATH" ]; then
    echo "Error: Folder $DATA_PATH does not exist"
    exit 1
fi

find "$DATA_PATH" -type f -name "*.glb" | while IFS= read -r file; do
    echo "Processing file $file..."
    process_mesh "$file"
done

bash scripts/sanity_check.sh -d "$OUTPUT_DIR/frames" -ef 20 -ed 0 -esf 0