"""Registry for language processors."""

import immutabledict
from typing import Dict, Type, Optional, Callable, TypeVar

from ifeval.languages.language_processor import BaseLanguageProcessor

T = TypeVar('T', bound=BaseLanguageProcessor)

# ISO 639-1 codes to language names.
LANGUAGE_CODES = immutabledict.immutabledict({
    "en": "English",
    "es": "Spanish",
    "pt": "Portuguese",
    "ar": "Arabic",
    "hi": "Hindi",
    "fr": "French",
    "ru": "Russian",
    "de": "German",
    "ja": "Japanese",
    "it": "Italian",
    "bn": "Bengali",
    "uk": "Ukrainian",
    "th": "Thai",
    "ur": "Urdu",
    "ta": "Tamil",
    "te": "Telugu",
    "bg": "Bulgarian",
    "ko": "Korean",
    "pl": "Polish",
    "he": "Hebrew",
    "fa": "Persian",
    "vi": "Vietnamese",
    "ne": "Nepali",
    "sw": "Swahili",
    "kn": "Kannada",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ml": "Malayalam",
    "fi": "Finnish",
})


class LanguageRegistry:
    """Registry for language processors."""
    
    def __init__(self):
        """Initialize an empty registry."""
        self._processors: Dict[str, Type[BaseLanguageProcessor]] = {}
    
    def register(self, language_code: str) -> Callable[[Type[T]], Type[T]]:
        """Register a language processor.
        
        Args:
            language_code: The ISO 639-1 language code.
            
        Returns:
            A decorator function that registers the processor.
        """
        def decorator(cls: Type[T]) -> Type[T]:
            self._processors[language_code] = cls
            return cls
        return decorator
    
    def get_processor_class(self, language_code: str) -> Type[BaseLanguageProcessor]:
        """Get a language processor class by language code.
        
        Args:
            language_code: The ISO 639-1 language code.
            
        Returns:
            The language processor class.
            
        Raises:
            ValueError: If the language is not supported.
        """
        if language_code not in self._processors:
            raise ValueError(f"Unsupported language: {language_code}")
        return self._processors[language_code]
    
    def create_processor(self, language_code: str) -> BaseLanguageProcessor:
        """Create a language processor instance.
        
        Args:
            language_code: The ISO 639-1 language code.
            
        Returns:
            A language processor instance.
            
        Raises:
            ValueError: If the language is not supported.
        """
        return self.get_processor_class(language_code)()
    
    def get_language_name(self, language_code: str) -> str:
        """Get the name of a language.
        
        Args:
            language_code: The ISO 639-1 language code.
            
        Returns:
            The language name.
            
        Raises:
            ValueError: If the language is not supported.
        """
        if language_code not in LANGUAGE_CODES:
            raise ValueError(f"Unknown language code: {language_code}")
        return LANGUAGE_CODES[language_code]