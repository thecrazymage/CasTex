#!/bin/bash

FOLDERS_FILE="objaverse_eval/assets/objaverse_subset.txt"
TOTAL_FOLDERS=$(grep -c . "$FOLDERS_FILE")

START_TIME=$(date +%d-%m-%Y_%H-%M-%S)
RUN_NAME="objaverse_eval_${START_TIME}"

LOGS_DIR="logs/$RUN_NAME"
mkdir -p "$LOGS_DIR"

TEMP_RUN_DIR="logs/run_logs/$RUN_NAME"
mkdir -p "$TEMP_RUN_DIR"
mkdir -p "${TEMP_RUN_DIR}/scripts"
mkdir -p "${TEMP_RUN_DIR}/tasks"

BATCH_SIZE=25
BASE_DIR="objaverse_eval/objaverse_data/obj"
NUM_BATCHES=$(( (TOTAL_FOLDERS + BATCH_SIZE - 1) / BATCH_SIZE ))

for ((batch = 0; batch < NUM_BATCHES; batch++)); do
    START_INDEX=$((batch * BATCH_SIZE))
    END_INDEX=$((START_INDEX + BATCH_SIZE - 1))
    [ $END_INDEX -ge $TOTAL_FOLDERS ] && END_INDEX=$((TOTAL_FOLDERS - 1))

    echo "Creating batch number $batch"

    BATCH_SCRIPT="${TEMP_RUN_DIR}/scripts/batch_script_${batch}.sh"

    cat > $BATCH_SCRIPT <<EOT
#!/bin/bash
#SBATCH --job-name="eval_${batch}"
#SBATCH --output="${TEMP_RUN_DIR}/tasks/batch_output_${batch}.out"
#SBATCH --error="${TEMP_RUN_DIR}/tasks/batch_error_${batch}.err"
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --constraint="type_e"
#SBATCH --time=15:0:0

module load Python CUDA/12.4 gnu10
source deactivate
conda activate castex

FOLDERS_FILE="$FOLDERS_FILE"
START_INDEX=$START_INDEX
END_INDEX=$END_INDEX
BASE_DIR="$BASE_DIR"
LOGS_DIR="$LOGS_DIR"

mapfile -t FOLDERS < <(sed -n "\$((START_INDEX+1)),\$((END_INDEX+1))p" "\$FOLDERS_FILE")

total_in_batch=\${#FOLDERS[@]}
current_step=0

for FOLDER in "\${FOLDERS[@]}"; do

    ((current_step++))
    echo "------------------------------------------------------------------------------"
    echo "[Batch Progress: \$current_step/\$total_in_batch] Processing: \$FOLDER"
    echo "------------------------------------------------------------------------------"

    if [[ -d "\$LOGS_DIR/\$FOLDER" ]]; then
        rm -rf "\$LOGS_DIR/\$FOLDER"
    fi

    mesh_location="\$BASE_DIR/\$FOLDER/mesh.obj"
    
    python main.py \
        --log_dir "\$LOGS_DIR" \
        --mesh_location "\$mesh_location" \
        --prompt "\$FOLDER" \
        --objaverse_eval \
        --num_steps_i 500 \
        --num_steps_ii 1000 \
        --use_dir_embeddings \
        --guidance_scale_i 15 \
        --guidance_scale_ii 10
done
rm -f "$BATCH_SCRIPT"
EOT
    chmod +x $BATCH_SCRIPT
    sbatch -A proj_1683 $BATCH_SCRIPT
done