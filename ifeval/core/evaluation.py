"""Core evaluation logic for instruction following."""

import collections
import dataclasses
import json
import logging
import os
from typing import Dict, List, Optional, Sequence, Union, Any, Tuple, Set

from ifeval.core.instructions import BaseInstruction
from ifeval.core.registry import InstructionRegistry


@dataclasses.dataclass
class InputExample:
    """Input example for evaluation."""
    instruction_id_list: List[str]
    prompt: str
    kwargs: List[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
    """Output example from evaluation."""
    instruction_id_list: List[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: List[bool]


@dataclasses.dataclass
class PassAtKExample:
    """Pass@k analysis results for a single prompt."""
    instruction_id_list: List[str]
    prompt: str
    responses: List[str]
    pass_at_k_score_strict: float
    pass_at_k_score_loose: float


class Evaluator:
    """Main evaluator for instruction following."""
    
    def __init__(self, registry: InstructionRegistry):
        """Initialize an evaluator.
        
        Args:
            registry: The instruction registry to use.
        """
        self.registry = registry
    
    def test_instruction_following_strict(
        self, inp: InputExample, response: str
    ) -> OutputExample:
        """Test if a response strictly follows instructions.
        
        Args:
            inp: The input example.
            response: The model's response to evaluate.
            
        Returns:
            An OutputExample with the evaluation results.
        """
        instruction_list = inp.instruction_id_list
        is_following_list = []

        for index, instruction_id in enumerate(instruction_list):
            kwargs = inp.kwargs[index] or {}
            instruction = self.registry.create_instruction(instruction_id, **kwargs)

            if response.strip() and instruction.check_following(response):
                is_following_list.append(True)
            else:
                is_following_list.append(False)

        return OutputExample(
            instruction_id_list=inp.instruction_id_list,
            prompt=inp.prompt,
            response=response,
            follow_all_instructions=all(is_following_list),
            follow_instruction_list=is_following_list,
        )
    
    def test_instruction_following_loose(
        self, inp: InputExample, response: str
    ) -> OutputExample:
        """Test if a response loosely follows instructions.
        
        This is more lenient than strict evaluation, trying various modifications
        to the response to see if any variant follows the instructions.
        
        Args:
            inp: The input example.
            response: The model's response to evaluate.
            
        Returns:
            An OutputExample with the evaluation results.
        """
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
            kwargs = inp.kwargs[index] or {}
            instruction = self.registry.create_instruction(instruction_id, **kwargs)

            is_following = False
            for r in all_responses:
                if r.strip() and instruction.check_following(r):
                    is_following = True
                    break

            is_following_list.append(is_following)

        return OutputExample(
            instruction_id_list=inp.instruction_id_list,
            prompt=inp.prompt,
            response=response,
            follow_all_instructions=all(is_following_list),
            follow_instruction_list=is_following_list,
        )
    
    def _calculate_metrics(self, outputs: List[OutputExample]) -> Dict[str, Any]:
        """Calculate metrics from evaluation outputs.
        
        Args:
            outputs: List of OutputExample objects to calculate metrics for.
            
        Returns:
            Dictionary with calculated metrics.
        """
        if not outputs:
            return {
                "prompt_accuracy": 0.0,
                "instruction_accuracy": 0.0,
                "category_accuracy": {},
                "instruction_accuracy_by_type": {}
            }

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
                instruction_id_type = instruction_id.split(":")[0]
                tier0_total[instruction_id_type] += 1
                if followed_or_not:
                    tier0_correct[instruction_id_type] += 1

            for instruction_id, followed_or_not in zip(
                instruction_id_list, follow_instruction_list
            ):
                tier1_total[instruction_id] += 1
                if followed_or_not:
                    tier1_correct[instruction_id] += 1

        # Calculate accuracies
        prompt_accuracy = prompt_correct / prompt_total if prompt_total > 0 else 0
        instruction_accuracy = instruction_correct / instruction_total if instruction_total > 0 else 0
        
        # Calculate category accuracies
        category_accuracy = {}
        for instruction_id in sorted(tier0_total.keys()):
            category_accuracy[instruction_id] = tier0_correct[instruction_id] / tier0_total[instruction_id]
        
        # Calculate instruction type accuracies
        instruction_accuracy_by_type = {}
        for instruction_id in sorted(tier1_total.keys()):
            instruction_accuracy_by_type[instruction_id] = tier1_correct[instruction_id] / tier1_total[instruction_id]
        
        return {
            "prompt_accuracy": prompt_accuracy,
            "instruction_accuracy": instruction_accuracy,
            "category_accuracy": category_accuracy,
            "instruction_accuracy_by_type": instruction_accuracy_by_type,
            "prompt_total": prompt_total,
            "prompt_correct": prompt_correct,
            "instruction_total": instruction_total,
            "instruction_correct": instruction_correct
        }
    
    def evaluate(
        self, 
        input_examples: List[InputExample], 
        responses: Dict[str, str]
    ) -> Tuple[Dict[str, Any], Dict[str, List[OutputExample]]]:
        """Evaluate a set of inputs and responses.
        
        Args:
            input_examples: List of input examples to evaluate.
            responses: Dictionary mapping prompts to responses.
            
        Returns:
            A tuple of (report_dict, output_examples_dict) where:
            - report_dict: Dictionary with metrics for both strict and loose evaluation
            - output_examples_dict: Dictionary with outputs for both evaluation modes
        """
        all_outputs = {}  # Dictionary to store both strict and loose outputs
        report = {}
        
        for eval_function, result_key in [
            (self.test_instruction_following_strict, "eval_results_strict"),
            (self.test_instruction_following_loose, "eval_results_loose"),
        ]:
            logging.info(f"Generating {result_key}...")
            outputs = []
            for inp in input_examples:
                response = responses.get(inp.prompt)
                if response is None:
                    logging.warning(f"No response found for prompt: {inp.prompt[:50]}...")
                    continue
                outputs.append(eval_function(inp, response))
            
            if not outputs:
                logging.error("No outputs generated. Check if responses match prompts.")
                return {
                    "eval_results_strict": {}, 
                    "eval_results_loose": {}
                }, {
                    "eval_results_strict": [], 
                    "eval_results_loose": []
                }
                
            # Calculate accuracy and save outputs
            follow_all_instructions = [o.follow_all_instructions for o in outputs]
            accuracy = sum(follow_all_instructions) / len(outputs)
            logging.info(f"Accuracy: {accuracy:.4f}")
            
            # Store outputs in the dictionary
            all_outputs[result_key] = outputs
            
            # Calculate detailed metrics and add to report
            metrics = self._calculate_metrics(outputs)
            report[result_key] = metrics
        
        return report, all_outputs
    
    def print_report(self, report: Dict[str, Any]) -> None:
        """Print a report based on evaluation metrics.
        
        Args:
            report: The report dictionary to print.
        """
        for result_key, metrics in report.items():
            print("=" * 64)
            print(f"{result_key} Accuracy Scores:")
            print(f"Prompt-level accuracy: {metrics['prompt_accuracy']:.4f} ({metrics['prompt_correct']}/{metrics['prompt_total']})")
            print(f"Instruction-level accuracy: {metrics['instruction_accuracy']:.4f} ({metrics['instruction_correct']}/{metrics['instruction_total']})")

            print("\nCategory-level accuracies:")
            for category, accuracy in metrics['category_accuracy'].items():
                print(f"  {category}: {accuracy:.4f}")

            print("\nInstruction-type accuracies:")
            for instruction_type, accuracy in metrics['instruction_accuracy_by_type'].items():
                print(f"  {instruction_type}: {accuracy:.4f}")
            print()

    def evaluate_pass_at_k_hard(
        self,
        input_examples: List[InputExample],
        responses: Dict[str, List[str]],
    ) -> Tuple[Dict[str, Dict[str, float]], List[PassAtKExample]]:
        """Hard estimator of pass@k: treats k as the number of provided responses per prompt.

        Args:
            input_examples: List of InputExample objects.
            responses: Mapping from prompt to list of responses.

        Returns:
            A tuple of (report, outputs) where:
                report: Dict with 'eval_results_strict' and 'eval_results_loose' keys containing prompt and instruction accuracy.
                outputs: List of PassAtKExample objects with per-prompt analysis.
        """
        strict_prompt_total = strict_prompt_correct = 0.0
        loose_prompt_total = loose_prompt_correct = 0.0
        strict_inst_total = strict_inst_correct = 0.0
        loose_inst_total = loose_inst_correct = 0.0
        outputs: List[PassAtKExample] = []

        for inp in input_examples:
            resp_list = responses.get(inp.prompt)
            if resp_list is None:
                continue
            # Count per-instruction correct responses
            c_strict: List[int] = [0] * len(inp.instruction_id_list)
            c_loose: List[int] = [0] * len(inp.instruction_id_list)
            passed_strict = passed_loose = False

            for r in resp_list:
                out_s = self.test_instruction_following_strict(inp, r)
                if out_s.follow_all_instructions:
                    passed_strict = True
                for idx, ok in enumerate(out_s.follow_instruction_list):
                    if ok:
                        c_strict[idx] += 1

                out_l = self.test_instruction_following_loose(inp, r)
                if out_l.follow_all_instructions:
                    passed_loose = True
                for idx, ok in enumerate(out_l.follow_instruction_list):
                    if ok:
                        c_loose[idx] += 1

            score_strict = 1.0 if passed_strict else 0.0
            score_loose = 1.0 if passed_loose else 0.0

            outputs.append(
                PassAtKExample(
                    instruction_id_list=inp.instruction_id_list,
                    prompt=inp.prompt,
                    responses=resp_list,
                    pass_at_k_score_strict=score_strict,
                    pass_at_k_score_loose=score_loose,
                )
            )

            strict_prompt_total += 1
            strict_prompt_correct += score_strict
            loose_prompt_total += 1
            loose_prompt_correct += score_loose

            for idx in range(len(inp.instruction_id_list)):
                strict_inst_total += 1
                if c_strict[idx] > 0:
                    strict_inst_correct += 1
                loose_inst_total += 1
                if c_loose[idx] > 0:
                    loose_inst_correct += 1

        metrics_strict = {
            "prompt_accuracy": strict_prompt_correct / strict_prompt_total if strict_prompt_total else 0.0,
            "instruction_accuracy": strict_inst_correct / strict_inst_total if strict_inst_total else 0.0,
        }
        metrics_loose = {
            "prompt_accuracy": loose_prompt_correct / loose_prompt_total if loose_prompt_total else 0.0,
            "instruction_accuracy": loose_inst_correct / loose_inst_total if loose_inst_total else 0.0,
        }
        report = {"eval_results_strict": metrics_strict, "eval_results_loose": metrics_loose}
        return report, outputs

    def evaluate_pass_at_k(
        self,
        input_examples: List[InputExample],
        responses: Dict[str, List[str]],
        k: int,
    ) -> Tuple[Dict[str, Dict[str, float]], List[PassAtKExample]]:
        r"""Smooth estimator of pass@k using the numerically stable formula:

            pass@k = 1 - \prod_{i=n-c+1}^n (1 - k / i)

        Args:
            input_examples: List of InputExample objects.
            responses: Mapping from prompt to list of responses.
            k: Number of samples to consider in pass@k.

        Returns:
            A tuple of (report, outputs) where:
                report: Dict with 'eval_results_strict' and 'eval_results_loose' keys containing prompt and instruction accuracy.
                outputs: List of PassAtKExample objects with per-prompt analysis.
        """
        strict_prompt_total = strict_prompt_correct = 0.0
        loose_prompt_total = loose_prompt_correct = 0.0
        strict_inst_total = strict_inst_correct = 0.0
        loose_inst_total = loose_inst_correct = 0.0
        outputs: List[PassAtKExample] = []

        for inp in input_examples:
            resp_list = responses.get(inp.prompt)
            if resp_list is None:
                continue
            n = len(resp_list)
            c_strict: List[int] = [0] * len(inp.instruction_id_list)
            c_loose: List[int] = [0] * len(inp.instruction_id_list)
            for r in resp_list:
                out_s = self.test_instruction_following_strict(inp, r)
                for idx, ok in enumerate(out_s.follow_instruction_list):
                    if ok:
                        c_strict[idx] += 1
                out_l = self.test_instruction_following_loose(inp, r)
                for idx, ok in enumerate(out_l.follow_instruction_list):
                    if ok:
                        c_loose[idx] += 1
            # Compute pass@k scores per prompt
            pass_strict = pass_at_k(
                n,
                sum(1 for r in resp_list if self.test_instruction_following_strict(inp, r).follow_all_instructions),
                k,
            )
            pass_loose = pass_at_k(
                n,
                sum(1 for r in resp_list if self.test_instruction_following_loose(inp, r).follow_all_instructions),
                k,
            )

            outputs.append(
                PassAtKExample(
                    instruction_id_list=inp.instruction_id_list,
                    prompt=inp.prompt,
                    responses=resp_list,
                    pass_at_k_score_strict=pass_strict,
                    pass_at_k_score_loose=pass_loose,
                )
            )
            strict_prompt_total += 1
            strict_prompt_correct += pass_strict
            loose_prompt_total += 1
            loose_prompt_correct += pass_loose
            for idx in range(len(inp.instruction_id_list)):
                strict_inst_total += 1
                strict_inst_correct += pass_at_k(n, c_strict[idx], k)
                loose_inst_total += 1
                loose_inst_correct += pass_at_k(n, c_loose[idx], k)

        metrics_strict = {
            "prompt_accuracy": strict_prompt_correct / strict_prompt_total if strict_prompt_total else 0.0,
            "instruction_accuracy": strict_inst_correct / strict_inst_total if strict_inst_total else 0.0,
        }
        metrics_loose = {
            "prompt_accuracy": loose_prompt_correct / loose_prompt_total if loose_prompt_total else 0.0,
            "instruction_accuracy": loose_inst_correct / loose_inst_total if loose_inst_total else 0.0,
        }
        report = {"eval_results_strict": metrics_strict, "eval_results_loose": metrics_loose}
        return report, outputs

def pass_at_k(n: int, c: int, k: int) -> float:
    r"""
    Numerically stable pass@k estimator:

        pass@k = 1 - \prod_{i=n-c+1}^n (1 - k / i)

    Args:
        n: Total number of sampled responses per prompt.
        c: Number of correct responses (following all instructions).
        k: Number of samples to consider.

    Returns:
        Estimated pass@k value.
    """
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod
