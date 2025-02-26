"""Language-specific implementations for instruction following evaluation."""

from ifeval.languages.language_processor import BaseLanguageProcessor
from ifeval.languages.language_registry import LanguageRegistry
from ifeval.languages.en.processor import EnglishProcessor

__all__ = [
    'BaseLanguageProcessor',
    'LanguageRegistry',
    'EnglishProcessor'
]