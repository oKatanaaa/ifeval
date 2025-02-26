"""English language processor implementation."""

import functools
import re
from typing import List

import langdetect
import nltk
from nltk.tokenize import RegexpTokenizer

from ifeval.languages.language_processor import BaseLanguageProcessor
from ifeval.languages.language_registry import LanguageRegistry

# Patterns for sentence splitting
_ALPHABETS = "([A-Za-z])"
_PREFIXES = "(Mr|St|Mrs|Ms|Dr)[.]"
_SUFFIXES = "(Inc|Ltd|Jr|Sr|Co)"
_STARTERS = r"(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
_ACRONYMS = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
_WEBSITES = "[.](com|net|org|io|gov|edu|me)"
_DIGITS = "([0-9])"
_MULTIPLE_DOTS = r"\.{2,}"


class EnglishProcessor(BaseLanguageProcessor):
    """English language processor implementation."""
    
    def detect_language(self, text: str) -> str:
        """Detect if text is in English.
        
        Args:
            text: Text to check.
            
        Returns:
            Language code ('en' for English).
        """
        try:
            return langdetect.detect(text)
        except langdetect.LangDetectException:
            # Default to English if detection fails
            return "en"
    
    @functools.lru_cache(maxsize=None)
    def _get_sentence_tokenizer(self):
        """Get NLTK sentence tokenizer, with caching."""
        return nltk.data.load("nltk:tokenizers/punkt/english.pickle")
    
    def count_sentences(self, text: str) -> int:
        """Count the number of sentences in text.
        
        Args:
            text: Text to count sentences in.
            
        Returns:
            Number of sentences.
        """
        tokenizer = self._get_sentence_tokenizer()
        sentences = tokenizer.tokenize(text)
        return len(sentences)
    
    def count_words(self, text: str) -> int:
        """Count the number of words in text.
        
        Args:
            text: Text to count words in.
            
        Returns:
            Number of words.
        """
        tokenizer = RegexpTokenizer(r"\w+")
        tokens = tokenizer.tokenize(text)
        return len(tokens)
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences.
        
        Args:
            text: Text to split.
            
        Returns:
            List of sentences.
        """
        text = " " + text + "  "
        text = text.replace("\n", " ")
        text = re.sub(_PREFIXES, "\\1<prd>", text)
        text = re.sub(_WEBSITES, "<prd>\\1", text)
        text = re.sub(_DIGITS + "[.]" + _DIGITS, "\\1<prd>\\2", text)
        text = re.sub(
            _MULTIPLE_DOTS,
            lambda match: "<prd>" * len(match.group(0)) + "<stop>",
            text,
        )
        if "Ph.D" in text:
            text = text.replace("Ph.D.", "Ph<prd>D<prd>")
        text = re.sub(r"\s" + _ALPHABETS + "[.] ", " \\1<prd> ", text)
        text = re.sub(_ACRONYMS + " " + _STARTERS, "\\1<stop> \\2", text)
        text = re.sub(
            _ALPHABETS + "[.]" + _ALPHABETS + "[.]" + _ALPHABETS + "[.]",
            "\\1<prd>\\2<prd>\\3<prd>",
            text,
        )
        text = re.sub(_ALPHABETS + "[.]" + _ALPHABETS + "[.]", "\\1<prd>\\2<prd>", text)
        text = re.sub(" " + _SUFFIXES + "[.] " + _STARTERS, " \\1<stop> \\2", text)
        text = re.sub(" " + _SUFFIXES + "[.]", " \\1<prd>", text)
        text = re.sub(" " + _ALPHABETS + "[.]", " \\1<prd>", text)
        if "\"" in text:
            text = text.replace("."", "".")
        if '"' in text:
            text = text.replace('."', '".')
        if "!" in text:
            text = text.replace('!"', '"!')
        if "?" in text:
            text = text.replace('?"', '"?')
        text = text.replace(".", ".<stop>")
        text = text.replace("?", "?<stop>")
        text = text.replace("!", "!<stop>")
        text = text.replace("<prd>", ".")
        sentences = text.split("<stop>")
        sentences = [s.strip() for s in sentences]
        if sentences and not sentences[-1]:
            sentences = sentences[:-1]
        return sentences
    
    def lemmatize(self, text: str) -> str:
        """Lemmatize text (simplified version).
        
        For English, we'll just lowercase everything for basic matching.
        A more complete implementation would use NLTK's lemmatizer.
        
        Args:
            text: Text to lemmatize.
            
        Returns:
            Lemmatized text.
        """
        return text.lower()


# Register the processor
language_registry = LanguageRegistry()
language_registry.register("en")(EnglishProcessor)