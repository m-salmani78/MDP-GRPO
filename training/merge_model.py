import argparse
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Merge PEFT adapter with base model.")
    parser.add_argument(
        "--adapter_dir",
        type=str,
        required=True,
        help="Path to the adapter checkpoint directory."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Path to save the merged model."
    )
    args = parser.parse_args()

    adapter_dir = args.adapter_dir
    out_dir = args.out_dir

    # Load adapter model and merge
    model = AutoPeftModelForCausalLM.from_pretrained(adapter_dir, device_map="cpu")
    merged = model.merge_and_unload()
    merged.save_pretrained(out_dir)

    # Save tokenizer
    tok = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    tok.save_pretrained(out_dir)

    print("Merged model saved to:", out_dir)

if __name__ == "__main__":
    main()
