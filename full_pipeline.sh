#!/bin/bash
set -euo pipefail

# ==============================
# Configuration constants
# ==============================
CUDA=1
MODEL_NAME="google/gemma-2-2b-it"
TRAIN_FILE="./data/rlhf_train.jsonl"
METHOD="MDP-GRPO"
RES_MODEL_DIR="./results/models/$MODEL_NAME/$METHOD"
RES_DATA_DIR="./results/output"
RES_PATH="${RES_DATA_DIR}/${MODEL_NAME}-$METHOD.jsonl"
CHECKED_NAME="${MODEL_NAME}-$METHOD"
GPU_MEMORY_UTILIZATION=0.9

# ==============================
# 1. Train
# ==============================
echo $METHOD
export CUDA_VISIBLE_DEVICES=$CUDA
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

python training/train.py \
  --model_name_or_path "$MODEL_NAME" \
  --train_file "$TRAIN_FILE" \
  --output_dir "$RES_MODEL_DIR" \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --num_train_epochs 1.0 \
  --temperature 1.0 \
  --learning_rate 1e-5 \
  --mlflow_experiment "Gemma" \
  --run_name "$METHOD" \
  --num_generations 4 \
  --DA_ALPHA 0.2 \
  --DA_GOAL_MU_MODE "max_half_and_group_mean" \
  --DO_PROSPECT True \
  --prospect_lambda_pos 2.0 \
  --prospect_lambda_neg 1.25 \
  --beta_by_adv_sign True \
  --prospect_applied_to_delta True \
  --temperature_list 0.1 0.4 0.7 1.0

# ==============================
# 2. Determine latest checkpoint
# ==============================
LATEST_CKPT_DIR=$(ls -d "$RES_MODEL_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
if [ -z "${LATEST_CKPT_DIR:-}" ]; then
  echo "❌ No checkpoints found in $RES_MODEL_DIR. Exiting." >&2
  exit 1
fi

MERGED_OUT_DIR="${LATEST_CKPT_DIR/checkpoint-*/merged}"

# ==============================
# 3. Merge adapter → full model
# ==============================
python training/merge_model.py \
  --adapter_dir "$LATEST_CKPT_DIR" \
  --out_dir "$MERGED_OUT_DIR"

# ==============================
# 4. Do inference
# ==============================
./scripts/do_inference.sh \
  --cuda "$CUDA" \
  --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
  --model_path "$MERGED_OUT_DIR" \
  --res_path "$RES_PATH"

# ==============================
# 5. Evaluate performance
# ==============================
./scripts/check.sh \
  --input_response_data "$RES_PATH" \
  --output_file_name "$CHECKED_NAME"

lm_eval  --model vllm \
  --model_args pretrained=$MERGED_OUT_DIR,dtype=auto,trust_remote_code=True,max_model_len=2048,gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
  --tasks ifeval,mmlu,ai2_arc \
  --batch_size auto \
  --output_path results/benchmark
