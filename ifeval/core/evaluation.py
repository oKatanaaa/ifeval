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
    key: int
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


class Evaluator:
    """Main evaluator for instruction following."""
    
    def __init__(self, registry: InstructionRegistry):
        """Initialize an evaluator.
        
        Args:
            registry: The instruction registry to use.
        """
        self.registry = registry
    
    def read_prompt_list(self, input_jsonl_filename: str) -> List[InputExample]:
        """Read prompts from a JSONL file.
        
        Args:
            input_jsonl_filename: Path to the input JSONL file.
            
        Returns:
            A list of InputExample objects.
        """
        inputs = []
        with open(input_jsonl_filename, "r", encoding="utf-8") as f:
            for line in f:
                example = json.loads(line)
                inputs.append(
                    InputExample(
                        key=example["key"],
                        instruction_id_list=example["instruction_id_list"],
                        prompt=example["prompt"],
                        kwargs=example["kwargs"],
                    )
                )
        return inputs
    
    def read_prompt_to_response_dict(self, input_jsonl_filename: str) -> Dict[str, str]:
        """Create a dictionary mapping prompts to responses.
        
        Args:
            input_jsonl_filename: Path to the input JSONL file.
            
        Returns:
            A dictionary mapping prompts to responses.
        """
        return_dict = {}
        with open(input_jsonl_filename, "r", encoding="utf-8") as f:
            for line in f:
                example = json.loads(line)
                return_dict[example["prompt"]] = example["response"]
        return return_dict
    
    def write_outputs(self, output_jsonl_filename: str, outputs: List[OutputExample]) -> None:
        """Write outputs to a JSONL file.
        
        Args:
            output_jsonl_filename: Path to the output JSONL file.
            outputs: List of OutputExample objects to write.
        """
        assert outputs
        with open(output_jsonl_filename, "w", encoding="utf-8") as f:
            for output in outputs:
                f.write(
                    json.dumps(
                        {
                            attr_name: getattr(output, attr_name)
                            for attr_name in [
                                name for name in dir(output) if not name.startswith("_")
                            ]
                        },
                        ensure_ascii=False,
                    )
                )
                f.write("\n")
    
    def test_instruction_following_strict(
        self, inp: InputExample, prompt_to_response: Dict[str, str]
    ) -> OutputExample:
        """Test if a response strictly follows instructions.
        
        Args:
            inp: The input example.
            prompt_to_response: Dictionary mapping prompts to responses.
            
        Returns:
            An OutputExample with the evaluation results.
        """
        response = prompt_to_response[inp.prompt]
        instruction_list = inp.instruction_id_list
        is_following_list = []

        for index, instruction_id in enumerate(instruction_list):
            instruction_cls = self.registry.get_instruction(instruction_id)
            instruction = instruction_cls(instruction_id)

            instruction.build_description(**inp.kwargs[index])
            args = instruction.get_instruction_args()
            if args and "prompt" in args:
                instruction.build_description(prompt=inp.prompt)

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
        self, inp: InputExample, prompt_to_response: Dict[str, str]
    ) -> OutputExample:
        """Test if a response loosely follows instructions.
        
        This is more lenient than strict evaluation, trying various modifications
        to the response to see if any variant follows the instructions.
        
        Args:
            inp: The input example.
            prompt_to_response: Dictionary mapping prompts to responses.
            
        Returns:
            An OutputExample with the evaluation results.
        """
        response = prompt_to_response[inp.prompt]
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
            instruction_cls = self.registry.get_instruction(instruction_id)
            instruction = instruction_cls(instruction_id)

            instruction.build_description(**inp.kwargs[index])
            args = instruction.get_instruction_args()
            if args and "prompt" in args:
                instruction.build_description(prompt=inp.prompt)

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
    
    def evaluate_dataset(
        self, 
        input_data_path: str, 
        input_response_path: str, 
        output_dir: str
    ) -> Tuple[float, float]:
        """Evaluate a dataset of prompts and responses.
        
        Args:
            input_data_path: Path to the input data JSONL file.
            input_response_path: Path to the input response JSONL file.
            output_dir: Directory to write output files to.
            
        Returns:
            A tuple of (strict_accuracy, loose_accuracy)
        """
        inputs = self.read_prompt_list(input_data_path)
        prompt_to_response = self.read_prompt_to_response_dict(input_response_path)
        
        results = {}
        
        for func, output_file_name in [
            (self.test_instruction_following_strict, "eval_results_strict"),
            (self.test_instruction_following_loose, "eval_results_loose"),
        ]:
            logging.info("Generating %s...", output_file_name)
            outputs = []
            for inp in inputs:
                outputs.append(func(inp, prompt_to_response))
            follow_all_instructions = [o.follow_all_instructions for o in outputs]
            accuracy = sum(follow_all_instructions) / len(outputs)
            logging.info("Accuracy: %f", accuracy)
            
            results[output_file_name] = accuracy

            output_file_name = os.path.join(output_dir, output_file_name + ".jsonl")
            self.write_outputs(output_file_name, outputs)
            logging.info("Generated: %s", output_file_name)

            # Print instruction following accuracy report
            print("=" * 64)
            print(f"{output_file_name} Accuracy Scores:")
            self.print_report(outputs)
        
        return results["eval_results_strict"], results["eval_results_loose"]
    
    def print_report(self, outputs: List[OutputExample]) -> None:
        """Print a report on accuracy scores.
        
        Args:
            outputs: List of OutputExample objects to report on.
        """
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

        print(f"prompt-level: {prompt_correct / prompt_total}")
        print(f"instruction-level: {instruction_correct / instruction_total}")
        print()
        for instruction_id in sorted(tier0_total.keys()):
            accuracy = tier0_correct[instruction_id] / tier0_total[instruction_id]
            print(f"{instruction_id} {accuracy}")
        print()
        for instruction_id in sorted(tier1_total.keys()):
            accuracy = tier1_correct[instruction_id] / tier1_total[instruction_id]
            print(f"{instruction_id} {accuracy}")