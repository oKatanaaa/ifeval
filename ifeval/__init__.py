"""Instruction Following Evaluation (IFEval) framework.

This package provides tools for evaluating how well language models follow instructions.
"""

from ifeval.core import (
    BaseInstruction,
    InstructionRegistry,
    Evaluator,
    InputExample,
    OutputExample
)
from ifeval.languages import (
    BaseLanguageProcessor,
    LanguageRegistry,
    EnglishProcessor,
    RussianProcessor,
    # Generic instructions
    PlaceholderChecker,
    BulletListChecker,
    HighlightSectionChecker,
    ParagraphChecker,
    JsonFormat,
    TwoResponsesChecker,
    TitleChecker,
    CommaChecker,
    QuotationChecker
)
from ifeval.utils import Config

from ifeval.utils.io import read_input_examples, read_responses, read_responses_list, write_outputs
from ifeval.utils.huggingface import get_default_dataset
from ifeval.languages.en.instructions import instruction_registry
from ifeval.languages.ru.instructions import instruction_registry as ru_instruction_registry

__version__ = "0.1.0"

__all__ = [
    # Core
    'BaseInstruction',
    'InstructionRegistry',
    'Evaluator',
    'InputExample',
    'OutputExample',
    
    # Languages
    'BaseLanguageProcessor',
    'LanguageRegistry',
    'EnglishProcessor',
    'RussianProcessor',
    
    # Generic Instructions
    'PlaceholderChecker',
    'BulletListChecker',
    'HighlightSectionChecker',
    'ParagraphChecker',
    'JsonFormat',
    'TwoResponsesChecker',
    'TitleChecker',
    'CommaChecker',
    'QuotationChecker',
    
    # Utils
    'Config',
    'read_input_examples',
    'read_responses',
    'read_responses_list',
    'write_outputs',
    'get_default_dataset',
    'instruction_registry',
    'ru_instruction_registry',
]
