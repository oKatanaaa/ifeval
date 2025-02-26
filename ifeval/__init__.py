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
]