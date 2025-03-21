"""Dataset utilities for instruction following evaluation."""

import json
from typing import List

from datasets import load_dataset

from ifeval.core.evaluation import InputExample
from ifeval.core import use_legacy_behavior


def get_default_dataset(language: str = "en") -> List[InputExample]:
    """
    Load the default dataset for a language from Hugging Face.
    
    Args:
        language: Language code, either "en" or "ru". Defaults to "en".
        
    Returns:
        List of InputExample objects.
        
    Raises:
        ValueError: If the dataset for the specified language cannot be loaded.
    """
    # if language == 'ru':
    #     language = 'ru' if use_legacy_behavior() else "ru_v2"
    try:
        # Load the dataset from HuggingFace
        data_files = {"test": f"{language}.jsonl"}
        dataset = load_dataset("kaleinaNyan/instruction-following-eval", data_files=data_files)
        
        # Default to the validation split
        dataset = dataset["test"]
        
        # Parse the examples
        examples = []
        for item in dataset:
            # Parse the content field which contains the JSON string
            content = json.loads(item["content"])
            
            examples.append(
                InputExample(
                    instruction_id_list=content["instruction_id_list"],
                    prompt=content["prompt"],
                    kwargs=content["kwargs"],
                )
            )
        
        return examples
    
    except Exception as e:
        raise ValueError(f"Failed to load dataset for {language}: {e}")