#!/usr/bin/env python3
# coding=utf-8
"""
Prepare training data for GRPO from the input JSONL format.

This script converts the input data (with prompts and instruction metadata)
into the format needed for GRPO training.

Input format (same as check.py):
{
  "prompt": "...",
  "instruction_id_list": ["keywords:existence", "length_constraints:number_words"],
  "kwargs": [{"keywords": ["word1", "word2"]}, {"num_words": 100}],
  "constraints": ["Include keywords...", "Your response should contain..."]
}

Output format (for GRPO):
Same as input - GRPO will use this directly and generate responses on-the-fly
"""

import argparse
import json
import logging
from pathlib import Path


def prepare_grpo_data(input_file: str, output_file: str):
    """
    Prepare GRPO training data.
    
    For GRPO, we don't need pre-generated responses. We just need to ensure
    the input data has the right format with prompts and instruction metadata.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
    """
    logging.info(f"Reading data from {input_file}")
    
    valid_count = 0
    output_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                example = json.loads(line)
                
                # Validate required fields
                if "prompt" not in example:
                    logging.warning(f"Line {line_num}: Missing 'prompt' field, skipping")
                    continue
                
                if "instruction_id_list" not in example:
                    logging.warning(f"Line {line_num}: Missing 'instruction_id_list' field, skipping")
                    continue
                
                if "kwargs" not in example:
                    logging.warning(f"Line {line_num}: Missing 'kwargs' field, skipping")
                    continue
                
                # Ensure kwargs is a list with same length as instruction_id_list
                if len(example["kwargs"]) != len(example["instruction_id_list"]):
                    logging.warning(
                        f"Line {line_num}: kwargs length ({len(example['kwargs'])}) "
                        f"!= instruction_id_list length ({len(example['instruction_id_list'])}), "
                        f"skipping"
                    )
                    continue
                
                # Create clean output example
                output_example = {
                    "prompt": example["prompt"],
                    "instruction_id_list": example["instruction_id_list"],
                    "kwargs": example["kwargs"],
                }
                
                # Optional: include constraints for reference
                if "constraints" in example:
                    output_example["constraints"] = example["constraints"]
                
                output_data.append(output_example)
                valid_count += 1
                
            except json.JSONDecodeError as e:
                logging.error(f"Line {line_num}: Invalid JSON - {e}")
                continue
            except Exception as e:
                logging.error(f"Line {line_num}: Error processing - {e}")
                continue
    
    # Write output
    logging.info(f"Writing {valid_count} examples to {output_file}")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in output_data:
            f.write(json.dumps(example, ensure_ascii=False))
            f.write('\n')
    
    logging.info(f"Successfully prepared {valid_count} examples for GRPO training")
    
    # Print statistics
    if output_data:
        num_instructions = [len(ex["instruction_id_list"]) for ex in output_data]
        avg_instructions = sum(num_instructions) / len(num_instructions)
        max_instructions = max(num_instructions)
        min_instructions = min(num_instructions)
        
        logging.info(f"Statistics:")
        logging.info(f"  Total examples: {valid_count}")
        logging.info(f"  Avg instructions per example: {avg_instructions:.2f}")
        logging.info(f"  Min instructions: {min_instructions}")
        logging.info(f"  Max instructions: {max_instructions}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data for GRPO from input JSONL"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input JSONL file with prompts and instruction metadata"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output JSONL file for GRPO training"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    prepare_grpo_data(args.input_file, args.output_file)


if __name__ == "__main__":
    main()

