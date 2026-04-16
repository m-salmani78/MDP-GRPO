#!/bin/bash

# Default values
CUDA_VISIBLE_DEVICES_DEFAULT=1
DATA_PATH="./src/data/data_test.jsonl"
BATCH_SIZE=128
MAX_MODEL_LEN=2048
MAX_TOKENS=1024
GPU_MEMORY_UTILIZATION=0.9
DTYPE="bfloat16"
NO_CHAT_TEMPLATE="--no_chat_template"

# Parse arguments
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_DEFAULT
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift ;;
        --res_path) RES_PATH="$2"; shift ;;
        --cuda) CUDA_VISIBLE_DEVICES="$2"; shift ;;
        --gpu_memory_utilization) GPU_MEMORY_UTILIZATION="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Check required arguments
if [ -z "$MODEL_PATH" ] || [ -z "$RES_PATH" ]; then
    echo "Usage: $0 --model_path <path> --res_path <path> [--cuda <device_ids>]"
    exit 1
fi

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Run inference
python ./src/do_inference.py \
  --data_path="$DATA_PATH" \
  --res_path="$RES_PATH" \
  --model_path="$MODEL_PATH" \
  --batch_size "$BATCH_SIZE" \
  --max_model_len "$MAX_MODEL_LEN" \
  --max_tokens "$MAX_TOKENS" \
  --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
  --dtype "$DTYPE"
