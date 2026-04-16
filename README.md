# MDP-GRPO: Dual-Anchor Group Relative Policy Optimization for Multi-Constraint Instruction Following

[![Github](https://img.shields.io/static/v1?logo=github&style=flat&label=github&message=m-salmani78/MDP-GRPO)](https://github.com/m-salmani78/MDP-GRPO)
[![Conference](https://img.shields.io/badge/ACL-2026-blue)](https://2026.aclweb.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Abstract

Reinforcement learning with verifiable rewards is ideal for multi-constraint instruction following, yet standard group-relative policy optimization (GRPO) becomes unstable under discrete, low-variance rewards that frequently produce homogeneous groups. 
We identify and formalize three pathologies of z-score group normalization in this regime: low-variance amplification, mean-centering blindness, and zero-variance collapse. 

To address them, we propose **MDP-GRPO**, which stabilizes learning through:
1. **M**ulti-temperature sampling to increase reward dispersion.
2. **D**ual-anchor advantages to restore gradients in homogeneous groups.
3. **P**rospect-theoretic shaping to bound updates and penalize violations based on Kahneman & Tversky's theory.
4. Asymmetric KL regularization. 

Evaluated on FollowBench, IFEval, and a curated multi-constraint dataset, MDP-GRPO outperforms standard GRPO, improving strict constraint satisfaction by up to 6.0% on Llama-3.2-3B. Our method also enables stable convergence with small group sizes while preserving general capabilities on MMLU and ARC.

## 📑 Table of Contents
- [Installation](#-installation)
- [Data Preparation](#-data-preparation)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Citation](#-citation)
- [Acknowledgments](#-acknowledgments)

## 🛠️ Installation
```bash
# Clone the repository
git clone https://github.com/m-salmani78/MDP-GRPO.git
cd MDP-GRPO

# Create a conda environment
conda create -n mdp-grpo python=3.10
conda activate mdp-grpo

# Install dependencies
pip install -r requirements.txt
```

## 🗂️ Data Preparation

Ensure your RLHF training data is formatted as a JSONL file. By default, the training pipeline expects the dataset to be located at `./training/data/rlhf_train.jsonl`. 

```bash
# Place your formatted data in the expected directory
mkdir -p ./training/data
# (Copy or download your rlhf_train.jsonl here)
```

## 🚀 Training

You can run the entire process (training, model merging, inference, and benchmarking) end-to-end using the provided pipeline script:

```bash
# Run the complete pipeline
bash full_pipeline.sh
```

### Step-by-step Training Process
If you prefer to run the steps manually, the pipeline executes the following stages:

**1. Train the LoRA Adapter (MDP-GRPO)**
By default, the script trains `google/gemma-2-2b-it`. *(Note: For Llama-3.2-3B mentioned in the paper, simply change the `MODEL_NAME` variable).*

```bash
python training/train.py \
  --model_name_or_path "google/gemma-2-2b-it" \
  --train_file "./training/data/rlhf_train.jsonl" \
  --output_dir "./results/models/google/gemma-2-2b-it/PT-GRPO" \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --num_train_epochs 1.0 \
  --temperature 1.0 \
  --learning_rate 1e-5 \
  --mlflow_experiment "Gemma" \
  --run_name "PT-GRPO" \
  --num_generations 8 \
  --DA_ALPHA 0 \
  --prospect_lambda_pos 2.0 \
  --prospect_lambda_neg 1.25
```

**2. Merge LoRA Weights**
After training, merge the generated LoRA adapter into the base model to prepare for vLLM inference:

```bash
python training/merge_model.py \
  --adapter_dir "./results/models/google/gemma-2-2b-it/PT-GRPO/checkpoint-[LATEST]" \
  --out_dir "./results/models/google/gemma-2-2b-it/PT-GRPO/merged"
```

## 📊 Evaluation

The evaluation stage consists of custom constraint checks and standardized benchmarking via `lm_eval`.

**1. Inference Generation**
Generate responses using vLLM:

```bash
./scripts/do_inference.sh \
  --cuda 1 \
  --gpu_memory_utilization 0.9 \
  --model_path "./results/models/google/gemma-2-2b-it/PT-GRPO/merged" \
  --res_path "./results/output/google/gemma-2-2b-it-PT-GRPO.jsonl"
```

**2. Custom Constraint Checking**
Evaluate the strict constraint satisfaction on the generated responses:

```bash
./scripts/check.sh \
  --input_response_data "./results/output/google/gemma-2-2b-it-PT-GRPO.jsonl" \
  --output_file_name "google/gemma-2-2b-it-PT-GRPO"
```

**3. General Capability Benchmarking (IFEval, MMLU, ARC)**
Run EleutherAI's `lm_eval` harness to ensure general capabilities are preserved:

```bash
lm_eval --model vllm \
  --model_args pretrained="./results/models/google/gemma-2-2b-it/PT-GRPO/merged",dtype=auto,trust_remote_code=True,max_model_len=2048,gpu_memory_utilization=0.9 \
  --tasks ifeval,mmlu,ai2_arc \
  --batch_size auto \
  --output_path results/benchmark
```

## 📝 Citation

If you find this repository or our paper useful, please consider citing our work:

```bibtex
@inproceedings{salmani2026mdpgrpo,
  title={MDP-GRPO: Dual-Anchor Group Relative Policy Optimization for Multi-Constraint Instruction Following},
  author={Salmani-Zarchi, Mohammad Mahdi and Rahimi, Zahra and Faili, Heshaam and Dousti, Mohammad Javad},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year={2026},
  publisher={Association for Computational Linguistics}
}
```

## 🤝 Acknowledgments
* This codebase builds upon [Standard GRPO Implementation / vLLM / TRL etc. - add links].
* We thank the reviewers at ACL 2026 for their valuable feedback.
