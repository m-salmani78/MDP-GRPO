# coding=utf-8
"""Reward function for GRPO training based on instruction following evaluation."""

import sys
import os
from typing import Dict, List, Any
import logging

# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import instructions_registry
except ImportError as e:
    logging.error(f"Failed to import instructions_registry: {e}")
    raise


class InstructionFollowingReward:
    """
    Reward function that evaluates how many constraints are followed in a response.
    
    The reward is calculated as the proportion of instructions followed (0.0 to 1.0).
    """
    
    def __init__(self, normalize: bool = True):
        """
        Initialize the reward function.
        
        Args:
            normalize: If True, return reward as proportion (0.0-1.0). 
                      If False, return raw count of followed instructions.
        """
        self.normalize = normalize
        
    def evaluate_single(
        self, 
        prompt: str,
        response: str,
        instruction_id_list: List[str],
        kwargs: List[Dict[str, Any]]
    ) -> float:
        """
        Evaluate a single response and return reward.
        
        Args:
            prompt: The input prompt
            response: The model's response to evaluate
            instruction_id_list: List of instruction IDs to check
            kwargs: List of kwargs for each instruction
            
        Returns:
            Reward score (proportion of instructions followed if normalize=True,
            otherwise count of instructions followed)
        """
        if not response or not response.strip():
            return 0.0
        
        if not instruction_id_list:
            # No instructions to check
            return 1.0 if self.normalize else 0.0
        
        followed_count = 0
        
        for index, instruction_id in enumerate(instruction_id_list):
            try:
                if instruction_id not in instructions_registry.INSTRUCTION_DICT:
                    logging.warning(f"Unknown instruction_id: {instruction_id}")
                    continue
                
                instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
                instruction = instruction_cls(instruction_id)
                
                # Build description with kwargs if available
                kw = kwargs[index] if index < len(kwargs) else {}
                desc_kwargs = dict(kw)

                # Only pass allowed keys to build_description
                args_keys = instruction.get_instruction_args_keys() or []
                filtered_kwargs = {k: v for k, v in desc_kwargs.items() if k in args_keys}

                # Add prompt if needed by the instruction
                if "prompt" in args_keys:
                    filtered_kwargs["prompt"] = prompt

                # Build the instruction description
                instruction.build_description(**filtered_kwargs)
                
                # Check if instruction is followed
                if instruction.check_following(response):
                    followed_count += 1
                    
            except Exception as e:
                logging.warning(f"Error checking instruction {instruction_id}: {e}")
                continue
        
        if self.normalize:
            return followed_count / len(instruction_id_list)
        else:
            return float(followed_count)
    
    def evaluate_batch(
        self,
        prompts: List[str],
        responses: List[str],
        instruction_id_lists: List[List[str]],
        kwargs_lists: List[List[Dict[str, Any]]]
    ) -> List[float]:
        """
        Evaluate a batch of responses.
        
        Args:
            prompts: List of input prompts
            responses: List of model responses
            instruction_id_lists: List of instruction ID lists
            kwargs_lists: List of kwargs lists
            
        Returns:
            List of reward scores
        """
        rewards = []
        for prompt, response, inst_ids, kw_list in zip(
            prompts, responses, instruction_id_lists, kwargs_lists
        ):
            reward = self.evaluate_single(prompt, response, inst_ids, kw_list)
            rewards.append(reward)
        
        return rewards
    
    def __call__(
        self,
        prompts: List[str],
        responses: List[str],
        instruction_id_lists: List[List[str]],
        kwargs_lists: List[List[Dict[str, Any]]]
    ) -> List[float]:
        """
        Callable interface for the reward function.
        """
        return self.evaluate_batch(prompts, responses, instruction_id_lists, kwargs_lists)

