#!/bin/bash

module load Python CUDA/12.4 gnu10
source deactivate
conda activate castex

if [ -z "$1" ]; then
    echo "Usage: bash run_metrics.sh <path_to_method_frames>"
    exit 1
fi

METHOD_PATH="$1"
GT_PATH="objaverse_eval/renders/ground_truth/frames"

INCEPTION_PATH="./objaverse_eval/assets/pt_inception-2015-12-05-6726825d.pth"
INCEPTION_URL="https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth"

if [ ! -f "$INCEPTION_PATH" ]; then
    echo "Downloading Inception weights..."
    wget -O "$INCEPTION_PATH" "$INCEPTION_URL"
else
    echo "Inception weights found at $INCEPTION_PATH"
fi

echo "Starting evaluation for: $METHOD_PATH"
echo "--------------------------------------------------"

python ./objaverse_eval/metrics/calculate_metrics.py \
    --method_path "$METHOD_PATH" \
    --gt_path "$GT_PATH" \
    --inception_path "$INCEPTION_PATH"

echo "--------------------------------------------------"
echo "Done!"