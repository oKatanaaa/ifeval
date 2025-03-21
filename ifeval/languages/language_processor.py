"""Base class for language processors."""

from abc import ABC, abstractmethod
from typing import List


class BaseLanguageProcessor(ABC):
    """Base class for language-specific processors."""
    
    @abstractmethod
    def count_sentences(self, text: str) -> int:
        """Count the number of sentences in a text.
        
        Args:
            text: The text to count sentences in.
            
        Returns:
            The number of sentences.
        """
        pass
    
    @abstractmethod
    def count_words(self, text: str) -> int:
        """Count the number of words in a text.
        
        Args:
            text: The text to count words in.
            
        Returns:
            The number of words.
        """
        pass
    
    @abstractmethod
    def split_into_sentences(self, text: str) -> List[str]:
        """Split a text into sentences.
        
        Args:
            text: The text to split.
            
        Returns:
            A list of sentences.
        """
        pass
    
    @abstractmethod
    def lemmatize(self, text: str) -> str:
        """Lemmatize a text.
        
        Args:
            text: The text to lemmatize.
            
        Returns:
            The lemmatized text.
        """
        pass
        
    @abstractmethod
    def word_tokenize(self, text: str) -> List[str]:
        """Tokenize text into words using language-specific rules.
        
        Args:
            text: The text to tokenize.
            
        Returns:
            A list of words.
        """
        pass