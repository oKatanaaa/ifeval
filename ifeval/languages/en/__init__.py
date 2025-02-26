"""English language implementation for instruction following evaluation."""

from ifeval.languages.en.processor import EnglishProcessor
from ifeval.languages.en.constants import (
    COMPARISON_RELATION, 
    CONSTRAINED_RESPONSE_OPTIONS,
    ENDING_OPTIONS,
    STARTER_OPTIONS,
    POSTSCRIPT_MARKER,
    SECTION_SPLITER
)

__all__ = [
    'EnglishProcessor',
    'COMPARISON_RELATION',
    'CONSTRAINED_RESPONSE_OPTIONS',
    'ENDING_OPTIONS',
    'STARTER_OPTIONS',
    'POSTSCRIPT_MARKER',
    'SECTION_SPLITER'
]