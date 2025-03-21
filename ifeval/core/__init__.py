"""Core module for instruction following evaluation."""

from ifeval.core.instructions import BaseInstruction
from ifeval.core.registry import InstructionRegistry
from ifeval.core.evaluation import Evaluator, InputExample, OutputExample

__all__ = [
    'BaseInstruction',
    'InstructionRegistry', 
    'Evaluator',
    'InputExample',
    'OutputExample'
]

_LEGACY = False

def legacy_behavior():
    return _LEGACY

def use_legacy_behavior(use: bool):
    global _LEGACY
    assert isinstance(use, bool)
    _LEGACY = use
