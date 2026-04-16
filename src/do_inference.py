import argparse
import json
import os
import warnings
from typing import List, Dict, Tuple

# -----------------------------------------------------------------------------
# Light, safe defaults
# -----------------------------------------------------------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=UserWarning)

# utils.readjsonl fallback (keeps your original import if present)
try:
    import utils  # must provide readjsonl(path)->List[dict]
except Exception:
    class _U:
        @staticmethod
        def readjsonl(p):
            out = []
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.append(json.loads(line))
            return out
    utils = _U()

# -----------------------------------------------------------------------------
# Try vLLM first
# -----------------------------------------------------------------------------
from vllm import LLM, SamplingParams

# Transformers fallback
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

DEFAULT_SYSTEM_MESSAGE = (
    "You are a helpful, respectful and honest assistant. Always answer as "
    "helpfully as possible while being safe. If the question is unclear or "
    "you don't know, say so."
)

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def check_gpu() -> Tuple[bool, int, str]:
    try:
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            print(f"[info] GPU: {name} (count={n})")
            # Enable TF32 for speed on Ampere+/Ada GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            return True, n, name
        print("[info] No GPU detected; will try CPU.")
        return False, 0, ""
    except Exception as e:
        print(f"[warn] GPU check failed: {e} -> using CPU")
        return False, 0, ""

def detect_model_type(model_path: str) -> str:
    m = model_path.lower()
    if "llama-3" in m or "llama3" in m: return "llama3"
    if "gemma-3" in m or "gemma3" in m: return "gemma"
    if "gemma" in m: return "gemma"
    if "llama-2" in m or "llama2" in m: return "llama2"
    if "mistral" in m or "mixtral" in m: return "mistral"
    if "qwen" in m: return "qwen"
    return "generic"

# -----------------------------------------------------------------------------
# Prompt formatting (chat template preferred)
# -----------------------------------------------------------------------------
def build_prompt_with_template(tokenizer, query, history, system_message):
    """
    Try chat template. If 'system' isn't supported, fold it into the user turn.
    """
    # First, try with system as-is
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    for u, a in history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": query})

    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        # If template refuses 'system', merge it into user and retry
        if "system role not supported" in str(e).lower() or "role not supported" in str(e).lower():
            merged_query = _fold_system_into_user(query, system_message)
            messages_no_system = []
            for u, a in history:
                messages_no_system.append({"role": "user", "content": u})
                messages_no_system.append({"role": "assistant", "content": a})
            messages_no_system.append({"role": "user", "content": merged_query})
            return tokenizer.apply_chat_template(messages_no_system, tokenize=False, add_generation_prompt=True)
        # Re-raise for outer fallback
        raise

def format_prompt_manual(model_path: str, query: str, history: List, system_message: str) -> str:
    mt = detect_model_type(model_path)
    if mt == "llama3":
        msgs = []
        if system_message: msgs.append(("system", system_message))
        for u, a in history: msgs += [("user", u), ("assistant", a)]
        msgs.append(("user", query))
        out = "<|begin_of_text|>"
        for role, content in msgs:
            out += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        out += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return out
    if mt == "gemma":
        out = "<bos>"
        if system_message: out += f"<start_of_turn>system\n{system_message}<end_of_turn>\n"
        for u, a in history:
            out += f"<start_of_turn>user\n{u}<end_of_turn>\n"
            out += f"<start_of_turn>model\n{a}<end_of_turn>\n"
        out += f"<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n"
        return out
    if mt == "llama2":
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        if not history:
            return f"<s>{B_INST} {B_SYS}{system_message}{E_SYS}{query} {E_INST} "
        out = ""
        for i, (u, a) in enumerate(history):
            if i == 0:
                out += f"<s>{B_INST} {B_SYS}{system_message}{E_SYS}{u} {E_INST} {a}</s>"
            else:
                out += f"<s>{B_INST} {u} {E_INST} {a}</s>"
        out += f"<s>{B_INST} {query} {E_INST} "
        return out
    if mt == "qwen":
        out = ""
        if system_message:
            out += f"<|im_start|>system\n{system_message}<|im_end|>\n"
        for u, a in history:
            out += f"<|im_start|>user\n{u}<|im_end|>\n"
            out += f"<|im_start|>assistant\n{a}<|im_end|>\n"
        out += f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        return out

    # generic
    prompt = f"System: {system_message}\n\n"
    for u, a in history:
        prompt += f"Human: {u}\nAssistant: {a}\n\n"
    prompt += f"Human: {query}\nAssistant: "
    return prompt

def build_prompt(model_path: str, query: str, tokenizer=None, use_chat_template=True,
                 history=None, system_message=None) -> str:
    history = history or []
    system_message = system_message or DEFAULT_SYSTEM_MESSAGE

    # Prefer chat template; if it throws, fall back to manual formatting
    if use_chat_template and tokenizer is not None and getattr(tokenizer, "chat_template", None):
        try:
            return build_prompt_with_template(tokenizer, query, history, system_message)
        except Exception:
            pass  # fall through to manual

    return format_prompt_manual(model_path, query, history, system_message)

# -----------------------------------------------------------------------------
# vLLM path
# -----------------------------------------------------------------------------
def setup_vllm_model(model_path: str,
                     tensor_parallel_size: int,
                     gpu_memory_utilization: float,
                     max_model_len: int,
                     dtype: str,
                     quantization: str,
                     swap_space: int,
                     trust_remote_code: bool) -> LLM:
    args = {
        "model": model_path,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": trust_remote_code,
        "dtype": dtype,                       # "half" | "bfloat16" | "float16" | "auto"
        "enforce_eager": False,
        "swap_space": swap_space,             # GB of CPU swap for KV if needed
    }
    if max_model_len:
        args["max_model_len"] = max_model_len
    if quantization:
        args["quantization"] = quantization   # e.g., "awq", "gptq", "bitsandbytes"

    print(f"[info] Loading with vLLM: {args}")
    return LLM(**args)

def vllm_generate(llm: LLM, prompts: List[str], temperature: float, top_p: float,
                  top_k: int, max_tokens: int, repetition_penalty: float) -> List[str]:
    sp = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k if (isinstance(top_k, int) and top_k > 0) else 40,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
    )
    outs = llm.generate(prompts, sp)
    res = []
    for o in outs:
        text = o.outputs[0].text if (o.outputs and len(o.outputs) > 0) else ""
        res.append(text.strip())
    return res

# -----------------------------------------------------------------------------
# Transformers fallback
# -----------------------------------------------------------------------------
class HFRunner:
    def __init__(self, model_path: str, dtype: str, trust_remote_code: bool):
        print(f"[info] Loading with Transformers: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        # Pad token hygiene
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        # dtype
        torch_dtype = {
            "auto": None,
            "half": torch.float16,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }.get(dtype or "auto", None)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype
        )
        if len(self.tokenizer) != self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.eval()
        # If it's a single-device model, keep a handle; otherwise let HF dispatch
        self.single_device = getattr(self.model, "device", None)
        self.has_map = hasattr(self.model, "hf_device_map") or hasattr(self.model, "device_map")

    def generate_batch(self, prompts: List[str], max_new_tokens: int,
                       temperature: float, top_p: float, top_k: int,
                       repetition_penalty: float) -> List[str]:
        out_texts = []
        for p in prompts:
            enc = self.tokenizer(p, return_tensors="pt", padding=False, truncation=False)
            inputs = {k: v for k, v in enc.items()}
            if self.single_device is not None and not self.has_map:
                inputs = {k: v.to(self.single_device) for k, v in inputs.items()}
            with torch.no_grad():
                gen_kwargs = dict(
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 0.1) if temperature > 0 else 1.0,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1
                )
                if top_k and top_k > 0:
                    gen_kwargs["top_k"] = top_k
                out = self.model.generate(**inputs, **gen_kwargs)
            inp_len = inputs["input_ids"].shape[1]
            seq = out[0]
            gen_part = seq[inp_len:] if seq.shape[0] > inp_len else seq
            text = self.tokenizer.decode(gen_part, skip_special_tokens=True)
            out_texts.append(text.strip())
        return out_texts

# -----------------------------------------------------------------------------
# Main logic
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser("Fast vLLM/Transformers runner")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--res_path", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--system_message", default="")
    # generation
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--max_tokens", type=int, default=512)
    # batching / memory
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--max_model_len", type=int, default=None)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--swap_space", type=int, default=4)
    # advanced
    ap.add_argument("--dtype", default="bfloat16", choices=["auto", "half", "float16", "bfloat16", "float32"])
    ap.add_argument("--quantization", default="", help="vLLM quantization: awq|gptq|bitsandbytes (if supported)")
    ap.add_argument("--no_chat_template", action="store_true")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--debug_cuda", action="store_true")
    args = ap.parse_args()

    if args.debug_cuda:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # paths
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(args.data_path)
    out_dir = os.path.dirname(args.res_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    system_msg = args.system_message or DEFAULT_SYSTEM_MESSAGE

    gpu_ok, gpu_count, _ = check_gpu()
    tp = min(max(1, args.tensor_parallel_size), gpu_count) if gpu_ok else 1
    if tp != args.tensor_parallel_size:
        print(f"[info] Adjusted tensor_parallel_size -> {tp}")

    # Load data
    data = utils.readjsonl(args.data_path)
    if not data:
        print("[warn] No samples found.")
        return
    total = len(data)
    print(f"[info] Loaded {total} samples")

    # Prepare model (prefer vLLM)
    runner = None
    using_vllm = False
    tokenizer_for_template = None

    if gpu_ok:
        is_gemma2 = "gemma-2" in args.model_path.lower() or "gemma2" in args.model_path.lower()
        if is_gemma2 and args.dtype in ("", "auto", "half", "float16"):
            args.dtype = "bfloat16"   # Gemma-2 requires bf16/float32
        try:
            llm = setup_vllm_model(
                model_path=args.model_path,
                tensor_parallel_size=tp,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                dtype=args.dtype,
                quantization=args.quantization,
                swap_space=args.swap_space,
                trust_remote_code=args.trust_remote_code,
            )
            using_vllm = True
            # Separate tokenizer for chat template if needed
            if not args.no_chat_template:
                try:
                    tokenizer_for_template = AutoTokenizer.from_pretrained(
                        args.model_path, trust_remote_code=args.trust_remote_code
                    )
                except Exception:
                    tokenizer_for_template = None
            print("[info] vLLM ready.")
        except Exception as e:
            print(f"[warn] vLLM init failed: {e}. Falling back to Transformers...")

    if not using_vllm:
        print("[warn] VLLM not available, falling back to Transformers...")
        runner = HFRunner(args.model_path, args.dtype, args.trust_remote_code)
        tokenizer_for_template = runner.tokenizer

    # Process in batches
    bs = max(1, args.batch_size)
    batches = (total + bs - 1) // bs

    with open(args.res_path, "w", encoding="utf-8") as fout:
        for b in range(batches):
            s = b * bs
            e = min((b + 1) * bs, total)
            batch = data[s:e]

            # Build prompts
            prompts = []
            for item in batch:
                q = item.get("prompt", item.get("input", "")) or ""
                if not q:
                    prompts.append("")
                    continue
                prompts.append(
                    build_prompt(
                        args.model_path,
                        q,
                        tokenizer=None if args.no_chat_template else tokenizer_for_template,
                        use_chat_template=not args.no_chat_template,
                        history=[],
                        system_message=system_msg,
                    )
                )

            # map valid indices
            idxs = [i for i, p in enumerate(prompts) if p]
            if not idxs:
                for item in batch:
                    item["response"] = "Error: Empty prompt"
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            valid_prompts = [prompts[i] for i in idxs]

            # Generate
            try:
                if using_vllm:
                    responses = vllm_generate(
                        llm,
                        valid_prompts,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        max_tokens=args.max_tokens,
                        repetition_penalty=1.0,
                    )
                else:
                    responses = runner.generate_batch(
                        valid_prompts,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        repetition_penalty=1.0,
                    )
            except Exception as e:
                # Write batch errors but keep going
                for item in batch:
                    item["response"] = f"Batch error: {e}"
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            # Attach responses back to items
            r = 0
            for i, item in enumerate(batch):
                if i in idxs:
                    item["response"] = responses[r]
                    r += 1
                else:
                    item["response"] = "Error: Empty prompt"
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"[info] Done. Saved to {args.res_path}")

if __name__ == "__main__":
    main()
