#!/bin/bash

module load Python CUDA/12.4 gnu10
source deactivate
conda activate castex

python objaverse_eval/objaverse_data_utils/download_objaverse_subset.py