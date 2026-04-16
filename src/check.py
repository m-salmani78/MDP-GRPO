"""Binary of evaluating instruction following. See README.md."""

import collections
import dataclasses
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Union, Any

from absl import app
from absl import flags
from absl import logging

import instructions


try:
    import instructions_registry
except ImportError as e:
    logging.error(f"Failed to import instructions_registry: {e}")
    logging.error("Make sure instructions_registry.py is in the same directory or in PYTHONPATH")
    sys.exit(1)


_INPUT_DATA = flags.DEFINE_string(
    "input_data", None, "path to input data", required=True
)

_INPUT_RESPONSE_DATA = flags.DEFINE_string(
    "input_response_data", None, "path to input response data", required=False
)

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "Output directory for inference and eval results.",
    required=True,
)

_OUTPUT_FILE_NAME = flags.DEFINE_string(
    "output_file_name",
    None,
    "Output file name for eval results.",
    required=True,
)

_EVALUATION_MODE = flags.DEFINE_enum(
    "evaluation_mode",
    "strict",
    ["strict", "loose"],
    "Evaluation mode: strict or loose"
)

_LANGUAGE = flags.DEFINE_string(
    "language", "en", "Language of instructions and responses ('en' or 'fa')."
)


@dataclasses.dataclass
class InputExample:
    # key: Optional[int] = None  # Made optional for compatibility
    constraints: List[str] = dataclasses.field(default_factory=list)
    instruction_id_list: List[str] = dataclasses.field(default_factory=list)
    prompt: str = ""
    kwargs: List[Dict[str, Optional[Union[str, int]]]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: List[str] = dataclasses.field(default_factory=list)
    prompt: str = ""
    response: str = ""
    constraints: List[str] = dataclasses.field(default_factory=list)
    follow_all_instructions: bool = False
    follow_instruction_list: List[bool] = dataclasses.field(default_factory=list)
    # key: Optional[int] = None  # Made optional for compatibility
    kwargs: List[Dict[str, Optional[Union[str, int]]]] = dataclasses.field(default_factory=list)


def validate_input_file(filename: str) -> bool:
    """Validate that input file exists and is readable."""
    if not os.path.exists(filename):
        logging.error(f"Input file does not exist: {filename}")
        return False
    
    if not os.access(filename, os.R_OK):
        logging.error(f"Input file is not readable: {filename}")
        return False
    
    return True


def read_prompt_list(input_jsonl_filename: str) -> List[InputExample]:
    """Read inputs from jsonl with better error handling."""
    if not validate_input_file(input_jsonl_filename):
        return []
    
    inputs = []
    try:
        with open(input_jsonl_filename, "r", encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                try:
                    example = json.loads(line)
                    
                    # Handle different input formats more flexibly
                    input_example = InputExample(
                        constraints=example.get("constraints", []),
                        instruction_id_list=example.get("instruction_id_list", []),
                        prompt=example.get("prompt", ""),
                        kwargs=example.get("kwargs", [])
                    )
                    
                    # Validate required fields
                    if not input_example.prompt:
                        logging.warning(f"Line {line_num}: Empty prompt, skipping")
                        continue
                    
                    if not input_example.instruction_id_list:
                        logging.warning(f"Line {line_num}: Empty instruction_id_list, skipping")
                        continue
                    
                    inputs.append(input_example)
                    
                except json.JSONDecodeError as e:
                    logging.error(f"Line {line_num}: Invalid JSON - {e}")
                    continue
                except Exception as e:
                    logging.error(f"Line {line_num}: Error processing - {e}")
                    continue
    
    except Exception as e:
        logging.error(f"Error reading file {input_jsonl_filename}: {e}")
        return []
    
    logging.info(f"Successfully read {len(inputs)} examples from {input_jsonl_filename}")
    return inputs


def write_outputs(output_jsonl_filename: str, outputs: List[OutputExample]) -> bool:
    """Writes outputs to jsonl with better error handling."""
    if not outputs:
        logging.error("No outputs to write")
        return False
    
    try:
        # Ensure output directory exists
        out_dir = os.path.dirname(output_jsonl_filename)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        
        with open(output_jsonl_filename, "w", encoding='utf-8') as f:
            for i, o in enumerate(outputs):
                try:
                    # Convert dataclass to dict more safely
                    output_dict = {
                        "instruction_id_list": o.instruction_id_list,
                        "prompt": o.prompt,
                        "response": o.response,
                        "constraints": o.constraints,
                        "follow_all_instructions": o.follow_all_instructions,
                        "follow_instruction_list": o.follow_instruction_list,
                        "kwargs": o.kwargs,
                    }
                    
                    f.write(json.dumps(output_dict, ensure_ascii=False))
                    f.write("\n")
                    
                except Exception as e:
                    logging.error(f"Error writing output {i}: {e}")
                    continue
        
        logging.info(f"Successfully wrote {len(outputs)} outputs to {output_jsonl_filename}")
        return True
        
    except Exception as e:
        logging.error(f"Error writing to {output_jsonl_filename}: {e}")
        return False


def test_instruction_following_strict(
    inp: InputExample,
    prompt_to_response: Dict[str, str],
) -> OutputExample:
    """Tests response to see if instructions are followed."""
    key = make_match_key(inp.prompt, inp.instruction_id_list, inp.kwargs)
    response = prompt_to_response.get(key, "")

    if not response:
        logging.warning(f"No response found for prompt: {inp.prompt[:100]}...")
    
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        try:
            if instruction_id not in instructions_registry.INSTRUCTION_DICT:
                logging.error(f"Unknown instruction_id: {instruction_id}")
                is_following_list.append(False)
                continue
            
            instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
            instruction = instruction_cls(instruction_id)

            # Build description with kwargs if available, using args keys
            kw = inp.kwargs[index] if index < len(inp.kwargs) else {}
            desc_kwargs = dict(kw)
            args_keys = instruction.get_instruction_args_keys() or []
            filtered_kwargs = {k: v for k, v in desc_kwargs.items() if k in args_keys}
            if "prompt" in args_keys:
                filtered_kwargs["prompt"] = inp.prompt
            # Build once
            instruction.build_description(**filtered_kwargs)

            if response.strip() and instruction.check_following(response):
                is_following_list.append(True)
            else:
                is_following_list.append(False)
                
        except Exception as e:
            logging.error(f"Error checking instruction {instruction_id}: {e}")
            is_following_list.append(False)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        constraints=inp.constraints,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
        kwargs=inp.kwargs,
    )


def test_instruction_following_loose(
    inp: InputExample,
    prompt_to_response: Dict[str, str],
) -> OutputExample:
    """Tests response for an upper bound for following instructions."""
    key = make_match_key(inp.prompt, inp.instruction_id_list, inp.kwargs)
    response = prompt_to_response.get(key, "")
    # response = prompt_to_response.get(inp.prompt, "")
    if not response:
        logging.warning(f"No response found for prompt: {inp.prompt[:100]}...")
    
    # Create variations of the response for loose evaluation
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        try:
            if instruction_id not in instructions_registry.INSTRUCTION_DICT:
                logging.error(f"Unknown instruction_id: {instruction_id}")
                is_following_list.append(False)
                continue
            
            instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
            instruction = instruction_cls(instruction_id)

            # Build description with kwargs if available, using args keys
            kw = inp.kwargs[index] if index < len(inp.kwargs) else {}
            desc_kwargs = dict(kw)
            args_keys = instruction.get_instruction_args_keys() or []
            filtered_kwargs = {k: v for k, v in desc_kwargs.items() if k in args_keys}
            if "prompt" in args_keys:
                filtered_kwargs["prompt"] = inp.prompt
            # Build once
            instruction.build_description(**filtered_kwargs)

            is_following = False
            for r in all_responses:
                if r.strip() and instruction.check_following(r):
                    is_following = True
                    break

            is_following_list.append(is_following)
            
        except Exception as e:
            logging.error(f"Error checking instruction {instruction_id}: {e}")
            is_following_list.append(False)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        constraints=inp.constraints,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
        kwargs=inp.kwargs,
    )

def make_match_key(prompt: str, instruction_id_list: List[str], kwargs: List[Dict[str, Any]]) -> str:
    # stable, JSON-serialized composite key
    return json.dumps(
        {"p": prompt, "i": instruction_id_list, "k": kwargs},
        ensure_ascii=False, sort_keys=True
    )

def read_prompt_to_response_dict(input_jsonl_filename: str) -> Dict[str, str]:
    """Creates dictionary matching prompt and response with better error handling."""
    if not validate_input_file(input_jsonl_filename):
        return {}
    
    return_dict = {}
    try:
        with open(input_jsonl_filename, "r", encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                try:
                    example = json.loads(line)
                    
                    # Handle different response formats
                    prompt = example.get("prompt", "")
                    instruction_id_list = example.get("instruction_id_list", [])
                    kwargs_list = example.get("kwargs", [])
                    response = example.get("response", example.get("output", ""))
                    
                    key = make_match_key(prompt, instruction_id_list, kwargs_list)
                    return_dict[key] = response
                    if not prompt:
                        logging.warning(f"Line {line_num}: Missing prompt")
                        continue
                    
                    if not response:
                        logging.warning(f"Line {line_num}: Missing response/output")
                        # Don't skip, use empty string
                        response = ""
                    
                    return_dict[prompt] = response
                    
                except json.JSONDecodeError as e:
                    logging.error(f"Line {line_num}: Invalid JSON - {e}")
                    continue
                except Exception as e:
                    logging.error(f"Line {line_num}: Error processing - {e}")
                    continue
    
    except Exception as e:
        logging.error(f"Error reading response file {input_jsonl_filename}: {e}")
        return {}
    
    logging.info(f"Successfully read {len(return_dict)} prompt-response pairs from {input_jsonl_filename}")
    return return_dict


def print_report(outputs: List[OutputExample]) -> Dict[str, float]:
    """Prints a report on accuracy scores and returns metrics."""
    if not outputs:
        logging.error("No outputs to report on")
        return {}

    prompt_total = 0
    prompt_correct = 0
    instruction_total = 0
    instruction_correct = 0

    tier0_total = collections.defaultdict(int)
    tier0_correct = collections.defaultdict(int)

    tier1_total = collections.defaultdict(int)
    tier1_correct = collections.defaultdict(int)

    for example in outputs:
        follow_instruction_list = example.follow_instruction_list
        instruction_id_list = example.instruction_id_list

        prompt_total += 1
        if all(follow_instruction_list):
            prompt_correct += 1

        instruction_total += len(instruction_id_list)
        instruction_correct += sum(follow_instruction_list)

        for instruction_id, followed_or_not in zip(
            instruction_id_list, follow_instruction_list
        ):
            base_instruction_id = instruction_id.split(":")[0]
            tier0_total[base_instruction_id] += 1
            if followed_or_not:
                tier0_correct[base_instruction_id] += 1

        for instruction_id, followed_or_not in zip(
            instruction_id_list, follow_instruction_list
        ):
            tier1_total[instruction_id] += 1
            if followed_or_not:
                tier1_correct[instruction_id] += 1

    # Difficulty buckets (by number of constraints: 1..5)
    diff_prompt_total = collections.defaultdict(int)
    diff_prompt_correct = collections.defaultdict(int)
    diff_instruction_total = collections.defaultdict(int)
    diff_instruction_correct = collections.defaultdict(int)

    for example in outputs:
        # Prefer explicit constraints length; fallback to instruction count if constraints missing
        level = len(example.constraints) if example.constraints is not None else 0
        if level == 0:
            level = len(example.instruction_id_list)
        # Clamp to 1..5 if out of expected bounds
        if level < 1:
            level = 1
        elif level > 6:
            level = 6

        # Prompt-level (all constraints followed)
        diff_prompt_total[level] += 1
        if all(example.follow_instruction_list):
            diff_prompt_correct[level] += 1

        # Instruction-level
        diff_instruction_total[level] += len(example.instruction_id_list)
        diff_instruction_correct[level] += sum(example.follow_instruction_list)

    # Calculate metrics
    prompt_accuracy = prompt_correct / prompt_total if prompt_total > 0 else 0
    instruction_accuracy = instruction_correct / instruction_total if instruction_total > 0 else 0

    for level in sorted(diff_prompt_total.keys()):
        acc = (
            diff_prompt_correct[level] / diff_prompt_total[level]
            if diff_prompt_total[level] > 0
            else 0
        )
    for level in sorted(diff_instruction_total.keys()):
        acc = (
            diff_instruction_correct[level] / diff_instruction_total[level]
            if diff_instruction_total[level] > 0
            else 0
        )

    # Build detailed metrics to return in JSON as well
    tier0_metrics = {}
    for instruction_id in sorted(tier0_total.keys()):
        total = tier0_total[instruction_id]
        correct = tier0_correct[instruction_id]
        acc = correct / total if total > 0 else 0
        tier0_metrics[instruction_id] = {
            "accuracy": acc,
            "correct": correct,
            "total": total,
        }

    tier1_metrics = {}
    for instruction_id in sorted(tier1_total.keys()):
        total = tier1_total[instruction_id]
        correct = tier1_correct[instruction_id]
        acc = correct / total if total > 0 else 0
        tier1_metrics[instruction_id] = {
            "accuracy": acc,
            "correct": correct,
            "total": total,
        }

    difficulty_metrics = {}
    for level in sorted(set(list(diff_prompt_total.keys()) + list(diff_instruction_total.keys()))):
        p_total = diff_prompt_total[level]
        p_correct = diff_prompt_correct[level]
        p_acc = p_correct / p_total if p_total > 0 else 0
        i_total = diff_instruction_total[level]
        i_correct = diff_instruction_correct[level]
        i_acc = i_correct / i_total if i_total > 0 else 0
        difficulty_metrics[str(level)] = {
            "prompt_accuracy": p_acc,
            "prompt_correct": p_correct,
            "prompt_total": p_total,
            "instruction_accuracy": i_acc,
            "instruction_correct": i_correct,
            "instruction_total": i_total,
        }

    return {
        "prompt_accuracy": prompt_accuracy,
        "instruction_accuracy": instruction_accuracy,
        "prompt_total": prompt_total,
        "prompt_correct": prompt_correct,
        "instruction_total": instruction_total,
        "instruction_correct": instruction_correct,
        "tier0": tier0_metrics,
        "tier1": tier1_metrics,
        "difficulty": difficulty_metrics,
    }


def check_followed(argv: Sequence[str]) -> None:
    """Main evaluation function."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # Set instruction language
    if _LANGUAGE.value:
        instructions.set_instruction_language(_LANGUAGE.value)
        logging.info(f"Instruction language set to: '{_LANGUAGE.value}'")

    # Validate inputs
    if not _INPUT_DATA.value:
        raise app.UsageError("--input_data is required")
    
    if not _INPUT_RESPONSE_DATA.value:
        raise app.UsageError("--input_response_data is required")
    
    if not _OUTPUT_DIR.value:
        raise app.UsageError("--output_dir is required")
    
    if not _OUTPUT_FILE_NAME.value:
        raise app.UsageError("--output_file_name is required")

    # Read inputs
    logging.info("Reading input data from %s", _INPUT_DATA.value)
    inputs = read_prompt_list(_INPUT_DATA.value)
    if not inputs:
        logging.error("No valid inputs found")
        return

    logging.info("Reading response data from %s", _INPUT_RESPONSE_DATA.value)
    prompt_to_response = read_prompt_to_response_dict(_INPUT_RESPONSE_DATA.value)
    if not prompt_to_response:
        logging.error("No valid responses found")
        return

    # Check for missing responses
    missing_responses = []
    for inp in inputs:
        k = make_match_key(inp.prompt, inp.instruction_id_list, inp.kwargs)
        if k not in prompt_to_response:
            missing_responses.append(inp.prompt[:100] + "...")
    
    if missing_responses:
        logging.warning(f"Found {len(missing_responses)} prompts without responses")
        if len(missing_responses) <= 5:
            for prompt in missing_responses:
                logging.warning(f"Missing response for: {prompt}")

    # Select evaluation function
    if _EVALUATION_MODE.value == "loose":
        func = test_instruction_following_loose
        logging.info("Using loose evaluation mode")
    else:
        func = test_instruction_following_strict
        logging.info("Using strict evaluation mode")

    output_file_name = _OUTPUT_FILE_NAME.value

    # Run evaluation
    logging.info("Generating %s...", output_file_name)
    outputs = []
    for i, inp in enumerate(inputs):
        try:
            result = func(inp, prompt_to_response)
            outputs.append(result)
            
            # Progress logging
            if (i + 1) % 100 == 0:
                logging.info(f"Processed {i + 1}/{len(inputs)} examples")
                
        except Exception as e:
            logging.error(f"Error processing example {i}: {e}")
            continue

    if not outputs:
        logging.error("No outputs generated")
        return

    # Calculate and log accuracy
    follow_all_instructions = [o.follow_all_instructions for o in outputs]
    accuracy = sum(follow_all_instructions) / len(outputs)
    logging.info("Overall accuracy: %f (%d/%d)", accuracy, sum(follow_all_instructions), len(outputs))

    # Write outputs
    output_file_path = os.path.join(_OUTPUT_DIR.value, output_file_name + ".jsonl")
    if write_outputs(output_file_path, outputs):
        logging.info("Generated: %s", output_file_path)
    else:
        logging.error("Failed to write outputs")
        return

    # Print detailed report
    metrics = print_report(outputs)
    
    # Optionally save metrics to JSON
    metrics_file = os.path.join(_OUTPUT_DIR.value, output_file_name + "_metrics.json")
    try:
        with open(metrics_file, "w", encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logging.info("Saved metrics to: %s", metrics_file)
    except Exception as e:
        logging.warning(f"Failed to save metrics: {e}")


if __name__ == "__main__":
    app.run(check_followed)