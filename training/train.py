# coding=utf-8
"""Native TRL GSPO trainer wired to the rule-based reward with dual-anchored shaping."""

import os
import sys
import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from collections import defaultdict

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, HfArgumentParser, TrainerCallback
from peft import LoraConfig
import mlflow
from trl import GRPOConfig, GRPOTrainer

# Local reward function (counts followed constraints)
sys.path.insert(0, os.path.dirname(__file__))
from reward_function import InstructionFollowingReward


# Global dictionary to store metrics across function calls
DUAL_ANCHOR_METRICS_BUFFER = defaultdict(list)
zero_A_num = 0
zero_z_num = 0


@dataclass
class GSPOTRLArguments:
    """Arguments for TRL-native GSPO training with dual-anchored advantages."""

    # Model
    model_name_or_path: str = "google/gemma-2-2b-it"
    hf_token: Optional[str] = ""
    trust_remote_code: bool = True

    # Data
    train_file: str = "./data/rlhf_train.jsonl"
    eval_file: Optional[str] = None

    # Generation / grouping
    num_generations: int = 8
    temperature: float = 1.0
    temperature_list: Optional[List[float]] = None
    top_p: float = 0.9
    top_k: Optional[int] = 50
    max_prompt_length: Optional[int] = 1024
    max_completion_length: Optional[int] = 1024

    # Training (GSPO-style)
    learning_rate: float = 1e-5
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    num_train_epochs: float = 1.0
    DA_ALPHA: float = 0.0
    DA_GOAL_MU_MODE: str = "max_half_and_group_mean"
    # Prospect (optional knobs, safe defaults)
    DO_PROSPECT: bool = False
    prospect_applied_to_delta: bool = False
    prospect_beta: float = 0.8
    prospect_lambda_pos: float = 1.25
    prospect_lambda_neg: float = 1.8

    # KL / GSPO
    beta: float = 0.01
    epsilon: float = 0.2 #TODO 3e-3
    epsilon_high: float = 0.2 #TODO 4e-3
    num_iterations: int = 1

    # Adaptive KL coefficient by advantage sign
    beta_pos: float = 0.01   # when Advantage >= 0
    beta_neg: float = 0.025   # when Advantage < 0
    beta_by_adv_sign: bool = False

    # GSPO-specific
    importance_sampling_level: str = "token" #TODO "sequence"

    # LoRA
    lora_r: int = 32
    lora_alpha: float = 64
    lora_dropout: float = 0.05

    # Logging / IO
    output_dir: str = "./results/grpo_trl"
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 1
    report_to: Optional[str] = "mlflow"
    mlflow_experiment: Optional[str] = None
    run_name: Optional[str] = None

    # VLLM
    use_vllm: bool = True
    vllm_mode: str = "colocate"

    # Misc
    seed: int = 42
    bf16: Optional[bool] = None


class DualAnchorMetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # avoid spam from non-zero ranks
        if hasattr(state, "is_world_process_zero") and not state.is_world_process_zero:
            return

        global DUAL_ANCHOR_METRICS_BUFFER, zero_A_num, zero_z_num
        if logs is None:
            logs = {}

        to_log = {}

        # Aggregate buffered metrics you filled in compute_rewards(...)
        for key, values in DUAL_ANCHOR_METRICS_BUFFER.items():
            if values:
                avg = sum(values) / len(values)
                if key in ("env_ssr_mean", "env_hsr_mean"):
                    # Comparable across methods: top-level train/ keys
                    to_log[f"train/{key.replace('_mean','')}"] = avg
                else:
                    # Keep diagnostics namespaced
                    to_log[f"dual_anchor/{key}"] = avg

        # Add counters
        to_log["dual_anchor/zero_z_num"] = zero_z_num
        to_log["dual_anchor/zero_A_num"] = zero_A_num

        # 1) Make them visible to Progress/Printer callbacks
        logs.update(to_log)

        # 2) Log explicitly to MLflow to avoid depending on callback order
        try:
            step = int(getattr(state, "global_step", 0))
            # MLflow likes flat dicts; use the same keys you want to see in the UI
            mlflow.log_metrics(to_log, step=step)
        except Exception as e:
            # Never crash training because logging failed
            logging.debug(f"[DualAnchorMetrics] mlflow.log_metrics failed: {e}")

        # Clear buffer for the next window
        DUAL_ANCHOR_METRICS_BUFFER.clear()

# def prospectify(A: torch.Tensor, beta: float, lambda_pos: float, lambda_neg: float) -> torch.Tensor:
#     A_abs = A.abs()
#     A_powered = torch.pow(A_abs + 1e-8, beta)
#     pos_val = lambda_pos * A_powered
#     neg_val = -lambda_neg * A_powered
#     return torch.where(A >= 0, pos_val, neg_val) #TODO

def prospectify(A: torch.Tensor, beta: float, lambda_pos: float, lambda_neg: float) -> torch.Tensor:
    pos = lambda_pos * torch.tanh(beta * A.clamp(min=0))
    neg = -lambda_neg * torch.tanh(beta * (-A).clamp(min=0))
    return torch.where(A >= 0, pos, neg)


def load_gspo_dataset(train_file: str, eval_file: Optional[str] = None):
    data_files = {"train": train_file}
    if eval_file:
        data_files["eval"] = eval_file

    dataset = load_dataset("json", data_files=data_files)

    required_cols = {"prompt"}
    for split, ds in dataset.items():
        missing = required_cols.difference(set(ds.column_names))
        if missing:
            raise ValueError(
                f"Missing required columns {missing} in split '{split}'. Got {ds.column_names}"
            )
    return dataset


def main():
    mlflow.set_tracking_uri("file:./mlruns")
    parser = HfArgumentParser(GSPOTRLArguments)
    args = parser.parse_args_into_dataclasses()[0]

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
    torch.manual_seed(args.seed)

    logging.info("=" * 80)
    if args.temperature_list:
        logging.info(f"🌡️  Multi-Temperature: {args.temperature_list}")
    else:
        logging.info(f"🌡️  Single Temperature: {args.temperature}")
    logging.info(f"Importance sampling level: {args.importance_sampling_level}")
    logging.info("=" * 80)

    logging.info(f"Loading tokenizer from {args.model_name_or_path}")
    tok_kwargs = {
        "use_fast": True,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.hf_token:
        tok_kwargs["token"] = args.hf_token
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if args.temperature_list and len(args.temperature_list) != args.num_generations:
        raise ValueError(
            f"temperature_list length ({len(args.temperature_list)}) must equal "
            f"num_generations ({args.num_generations})"
        )
    dataset = load_gspo_dataset(args.train_file, args.eval_file)
    reward_fn = InstructionFollowingReward(normalize=True)

    def compute_rewards(
        completions: List[Any],
        prompts: List[str] = None,
        instruction_id_list: List[List[str]] = None,
        kwargs: List[List[Dict[str, Any]]] = None,
        **_,
    ) -> List[float]:
        """Returns shaped sequence-level advantages (dual-anchored) per completion for GSPO."""
        
        global DUAL_ANCHOR_METRICS_BUFFER

        def to_text(c):
            if isinstance(c, list) and c and isinstance(c[0], dict) and "content" in c[0]:
                return "".join(m.get("content", "") for m in c)
            return c

        samples = [to_text(c) for c in completions]
        instr_lists = instruction_id_list if instruction_id_list is not None else [[] for _ in samples]
        kw = kwargs if kwargs is not None else [[] for _ in samples]
        return reward_fn(prompts, samples, instr_lists, kw)  # raw rewards

    # Autodetect bf16 if not set
    bf16 = args.bf16
    if bf16 is None:
        bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8

    if args.mlflow_experiment:
        mlflow.set_experiment(args.mlflow_experiment)
    
    gspo_config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
        run_name=args.run_name if args.run_name else os.path.basename(args.output_dir),
        seed=args.seed,
        # dtype / perf
        bf16=bool(bf16),
        gradient_checkpointing=True,
        remove_unused_columns=False,
        # Generation params
        num_generations=args.num_generations,
        temperature=args.temperature,
        temperature_list=args.temperature_list,
        top_p=args.top_p,
        top_k=args.top_k,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        # GSPO loss / KL
        loss_type="grpo",
        beta=args.beta,
        epsilon=args.epsilon,
        epsilon_high=args.epsilon_high,
        num_iterations=args.num_iterations,
        importance_sampling_level=args.importance_sampling_level,
        steps_per_generation=4,
        # generation_kwargs can be added if you need extra sampling args
        # model_init_kwargs lets you forward torch_dtype, low_cpu_mem_usage, trust_remote_code, etc.
        model_init_kwargs={
            "torch_dtype": torch.bfloat16 if bf16 else None,
            "low_cpu_mem_usage": True,
            "trust_remote_code": args.trust_remote_code,
        },
        # # Use vllm for faster generation
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_gpu_memory_utilization=0.4,
    )
    # Ensure TRL won't re-normalize our shaped advantages:
    if args.DA_ALPHA > 0 or args.DO_PROSPECT:
        gspo_config.scale_rewards="none"
        logging.info("# Scale Rewards = 'none'")
    #Dual-Anchoring
    gspo_config.advantage_mode = "dual_anchor"
    gspo_config.dual_anchor_alpha = args.DA_ALPHA
    gspo_config.dual_anchor_reward_is_normalized = True  # since normalize=False
    gspo_config.dual_anchor_constraint_key = "instruction_id_list"
    gspo_config.dual_anchor_goal_mu_mode = args.DA_GOAL_MU_MODE
    #Prospect
    gspo_config.prospect_enable = args.DO_PROSPECT
    gspo_config.prospect_applied_to_delta = args.prospect_applied_to_delta
    gspo_config.prospect_beta = args.prospect_beta
    gspo_config.prospect_lambda_pos = args.prospect_lambda_pos
    gspo_config.prospect_lambda_neg = args.prospect_lambda_neg
    # Attach extra knobs onto the config object (no need to edit GRPOConfig dataclass)
    gspo_config.beta_by_adv_sign = args.beta_by_adv_sign
    gspo_config.beta_pos = args.beta_pos
    gspo_config.beta_neg = args.beta_neg


    if args.hf_token:
        gspo_config.model_init_kwargs["token"] = args.hf_token

    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    logging.info("Initializing GSPO Trainer with sequence-level importance sampling...")
    trainer = GRPOTrainer(
        model=args.model_name_or_path,
        reward_funcs=compute_rewards,
        args=gspo_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("eval"),
        processing_class=tokenizer,
        peft_config=peft_cfg,
        callbacks=[DualAnchorMetricsCallback()],
    )

    logging.info("Starting GSPO training with dual-anchored advantages...")
    trainer.train()

    logging.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logging.info(f"GSPO training complete. Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
