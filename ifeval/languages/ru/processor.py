"""Russian language processor implementation."""

import functools
import re
from typing import List

from ftlangdetect import detect
import nltk
from pymorphy2 import MorphAnalyzer

from ifeval.languages.language_processor import BaseLanguageProcessor
from ifeval.languages.language_registry import LanguageRegistry

# Initialize the morphological analyzer
morph = MorphAnalyzer()

# Patterns for sentence splitting
_ALPHABETS = "([А-Яа-яA-Za-z])"
_PREFIXES = "(г|гр|г-н|г-жа|д-р|к|к-т|м-р|п|с)[.]"
_SUFFIXES = "(ООО|ОАО|ЗАО|АО|ИП)"
_STARTERS = r"(Он\s|Она\s|Оно\s|Они\s|Их\s|Мы\s|Но\s|Однако\s|Что\s|Это\s|Тот\s|Та\s)"
_ACRONYMS = "([А-Я][.][А-Я][.](?:[А-Я][.])?)"
_WEBSITES = "[.](com|net|org|io|gov|edu|me|ru)"
_DIGITS = "([0-9])"
_MULTIPLE_DOTS = r"\.{2,}"

# Pattern for removing Latin characters and symbols
_LATIN_SYMBOLS_PATTERN = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"


class RussianProcessor(BaseLanguageProcessor):
    """Russian language processor implementation."""
    
    def detect_language(self, text: str) -> str:
        """Detect if text is in Russian.
        
        Args:
            text: Text to check.
            
        Returns:
            Language code ('ru' for Russian).
        """
        try:
            return detect(text=text.replace("\n", " "), low_memory=False)["lang"]
        except Exception:
            # Default to Russian if detection fails
            return "ru"
    
    @functools.lru_cache(maxsize=None)
    def _get_sentence_tokenizer(self):
        """Get NLTK sentence tokenizer, with caching."""
        return nltk.data.load("nltk:tokenizers/punkt/russian.pickle")
    
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
        tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
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
        if "к.т.н" in text.lower() or "д.т.н" in text.lower():
            text = text.replace("к.т.н.", "к<prd>т<prd>н<prd>")
            text = text.replace("д.т.н.", "д<prd>т<prd>н<prd>")
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
        """Lemmatize text using pymorphy2.
        
        Args:
            text: Text to lemmatize.
            
        Returns:
            Lemmatized text.
        """
        text = re.sub(_LATIN_SYMBOLS_PATTERN, ' ', text)
        tokens = []
        for token in text.split():
            token = token.strip()
            token = morph.normal_forms(token)[0]
            tokens.append(token)
        return ' '.join(tokens)


# Register the processor
language_registry = LanguageRegistry()
language_registry.register("ru")(RussianProcessor)