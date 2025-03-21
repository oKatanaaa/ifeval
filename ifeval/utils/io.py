"""I/O utilities for instruction following evaluation."""

import json
from typing import Dict, List, Union

from ifeval.core.evaluation import InputExample, OutputExample

def read_input_examples(input_data: Union[str, List[str], List[dict]]) -> List[InputExample]:
    """Read input examples from a JSONL file, list of JSON strings, or list of dictionaries.

    Args:
        input_data: Can be one of the following:
            - Path to a JSONL file (str)
            - List of JSON strings (List[str])
            - List of dictionaries (List[dict])

    Returns:
        A list of InputExample objects.
    """
    inputs = []
    if isinstance(input_data, str):
        with open(input_data, "r", encoding="utf-8") as f:
            for line in f:
                example = json.loads(line)
                inputs.append(
                    InputExample(
                        instruction_id_list=example["instruction_id_list"],
                        prompt=example["prompt"],
                        kwargs=example["kwargs"],
                    )
                )
    elif isinstance(input_data, list):
        for item in input_data:
            example = json.loads(item) if isinstance(item, str) else item
            inputs.append(
                InputExample(
                    instruction_id_list=example["instruction_id_list"],
                    prompt=example["prompt"],
                    kwargs=example["kwargs"],
                )
            )
    else:
        raise ValueError("Unsupported input type. Must be a filename (str), list of JSON strings, or list of dicts.")

    return inputs

def read_responses(input_jsonl_filename: str) -> Dict[str, str]:
    """Create a dictionary mapping prompts to responses from a JSONL file.
    
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

def write_outputs(output_jsonl_filename: str, outputs: List[OutputExample]) -> None:
    """Write outputs to a JSONL file.
    
    Args:
        output_jsonl_filename: Path to the output JSONL file.
        outputs: List of OutputExample objects to write.
    """
    if not outputs:
        return
        
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