"""English language instruction implementations."""

import collections
import json
import random
import re
import string
from typing import Dict, List, Optional, Sequence, Union, Set, Any

from absl import logging

import langdetect
import nltk

from ifeval.core.instructions import BaseInstruction
from ifeval.core.registry import InstructionRegistry
from ifeval.languages.en.constants import (
    COMPARISON_RELATION,
    CONSTRAINED_RESPONSE_OPTIONS
)
from ifeval.languages.en.processor import EnglishProcessor
from ifeval.languages.generic import (
    PlaceholderChecker,
    BulletListChecker,
    HighlightSectionChecker,
    ParagraphChecker,
    JsonFormat,
    TwoResponsesChecker,
    TitleChecker,
    CommaChecker,
    QuotationChecker,
    # Add newly moved classes
    SectionChecker,
    PostscriptChecker,
    RepeatPromptThenAnswer,
    EndChecker,
    LetterFrequencyChecker,
    ResponseLanguageChecker
)

# Create registry and processor instances
instruction_registry = InstructionRegistry()
processor = EnglishProcessor()

# Define instruction type prefixes for registry
_KEYWORD = "keywords:"
_LANGUAGE = "language:"
_LENGTH = "length_constraints:"
_CONTENT = "detectable_content:"
_FORMAT = "detectable_format:"
_MULTITURN = "multi-turn:"
_COMBINATION = "combination:"
_STARTEND = "startend:"
_CHANGE_CASES = "change_case:"
_PUNCTUATION = "punctuation:"

# Register generic instructions
instruction_registry.register(_CONTENT + "number_placeholders")(PlaceholderChecker)
instruction_registry.register(_FORMAT + "number_bullet_lists")(BulletListChecker)
instruction_registry.register(_FORMAT + "number_highlighted_sections")(HighlightSectionChecker)
instruction_registry.register(_LENGTH + "number_paragraphs")(ParagraphChecker)
instruction_registry.register(_FORMAT + "json_format")(JsonFormat)
instruction_registry.register(_COMBINATION + "two_responses")(TwoResponsesChecker)
instruction_registry.register(_FORMAT + "title")(TitleChecker)
instruction_registry.register(_PUNCTUATION + "no_comma")(CommaChecker)
instruction_registry.register(_STARTEND + "quotation")(QuotationChecker)
instruction_registry.register(_FORMAT + "multiple_sections")(SectionChecker)
instruction_registry.register(_CONTENT + "postscript")(PostscriptChecker)
instruction_registry.register(_COMBINATION + "repeat_prompt")(RepeatPromptThenAnswer)
instruction_registry.register(_STARTEND + "end_checker")(EndChecker)
instruction_registry.register(_KEYWORD + "letter_frequency")(LetterFrequencyChecker)
instruction_registry.register(_LANGUAGE + "response_language")(ResponseLanguageChecker)


# Language-specific instructions
@instruction_registry.register(_LENGTH + "number_sentences")
class NumberOfSentences(BaseInstruction):
    """Check if response contains specific number of sentences.
    
    Example description:
    "Your response should contain {relation} {num_sentences} sentences."
    """

    def __init__(self, num_sentences, relation):
        """Initialize the sentence number checker.
        
        Args:
            num_sentences: An integer specifying the number of sentences as a
                threshold.
            relation: A string in (`less than`, `at least`), defining the relational
                operator for comparison.
        """
        # The number of sentences as a threshold for comparison.
        self._num_sentences_threshold = num_sentences

        if relation not in COMPARISON_RELATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                f"{COMPARISON_RELATION}, but {relation} is given."
            )
        
        self._comparison_relation = relation

    def get_instruction_args(self):
        """Returns the keyword args of the instruction."""
        return {
            "num_sentences": self._num_sentences_threshold,
            "relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of the instruction."""
        return ["num_sentences", "relation"]

    def check_following(self, value):
        """Check if the number of sentences follows the instruction.
        
        Args:
            value: A string representing the response.
            
        Returns:
            True if the response follows the instruction.
        """
        num_sentences = processor.count_sentences(value)
        if self._comparison_relation == COMPARISON_RELATION[0]:  # less than
            return num_sentences < self._num_sentences_threshold
        elif self._comparison_relation == COMPARISON_RELATION[1]:  # at least
            return num_sentences >= self._num_sentences_threshold


@instruction_registry.register(_FORMAT + "constrained_response")
class ConstrainedResponseChecker(BaseInstruction):
    """Check if response contains one of the constrained options.
    
    Example description:
    "Answer with one of the following options: {response_options}"
    """

    def __init__(self):
        """Initialize the constrained response checker."""
        # A sequence of string(s) representing the options of the expected response.
        self._constrained_responses = CONSTRAINED_RESPONSE_OPTIONS

    def get_instruction_args(self):
        """Returns the keyword args of the instruction."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of the instruction."""
        return []

    def check_following(self, value):
        """Checks if the response matches the constrained options.
        
        Args:
            value: A string representing the response.
            
        Returns:
            True if the actual response contains one of the options.
        """
        value = value.strip()
        for constrained_response in self._constrained_responses:
            if constrained_response in value:
                return True
        return False


@instruction_registry.register(_KEYWORD + "existence")
class KeywordChecker(BaseInstruction):
    """Check if response contains specified keywords.
    
    Example description:
    "Include keywords {keywords} in the response."
    """

    def __init__(self, keywords):
        """Initialize the keyword checker.
        
        Args:
            keywords: A sequence of strings representing the keywords that are
                expected in the response.
        """
        self._keywords = sorted(keywords)

    def get_instruction_args(self):
        """Returns the keyword args of the instruction."""
        return {"keywords": self._keywords}

    def get_instruction_args_keys(self):
        """Returns the args keys of the instruction."""
        return ["keywords"]

    def check_following(self, value):
        """Check if the response contain the expected keywords."""
        for keyword in self._keywords:
            if not re.search(keyword, value, flags=re.IGNORECASE):
                return False
        return True


@instruction_registry.register(_KEYWORD + "frequency")
class KeywordFrequencyChecker(BaseInstruction):
    """Check if a keyword appears in the response with specified frequency.
    
    Example description:
    "In your response, the word {keyword} should appear {relation}
    {frequency} times."
    """

    def __init__(self, keyword, frequency, relation):
        """Initialize the keyword frequency checker.
        
        Args:
            keyword: A string representing a keyword that is expected in the response.
            frequency: An integer specifying the number of times `keyword` is expected
                to appear in the response.
            relation: A string in (`less than`, `at least`), defining the relational
                operator for comparison.
        """
        self._keyword = keyword.strip()
        self._frequency = frequency

        if relation not in COMPARISON_RELATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                f"{COMPARISON_RELATION}, but {relation} is given."
            )
        
        self._comparison_relation = relation

    def get_instruction_args(self):
        """Returns the keyword args of the instruction."""
        return {
            "keyword": self._keyword,
            "frequency": self._frequency,
            "relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of the instruction."""
        return ["keyword", "frequency", "relation"]

    def check_following(self, value):
        """Checks if the response contain the keyword with required frequency."""
        actual_occurrences = len(
            re.findall(self._keyword, value, flags=re.IGNORECASE)
        )

        if self._comparison_relation == COMPARISON_RELATION[0]:  # less than
            return actual_occurrences < self._frequency
        elif self._comparison_relation == COMPARISON_RELATION[1]:  # at least
            return actual_occurrences >= self._frequency


@instruction_registry.register(_LENGTH + "number_words")
class NumberOfWords(BaseInstruction):
    """Check if response contains specified number of words.
    
    Example description:
    "Answer with {relation} {num_words} words."
    """

    def __init__(self, num_words, relation):
        """Initialize the word count checker.
        
        Args:
            num_words: An integer specifying the number of words contained in the
                response.
            relation: A string in (`less than`, `at least`), defining the relational
                operator for comparison.
        """
        self._num_words = num_words

        if relation not in COMPARISON_RELATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                f"{COMPARISON_RELATION}, but {relation} is given."
            )
        
        self._comparison_relation = relation

    def get_instruction_args(self):
        """Returns the keyword args of the instruction."""
        return {"num_words": self._num_words, "relation": self._comparison_relation}

    def get_instruction_args_keys(self):
        """Returns the args keys of the instruction."""
        return ["num_words", "relation"]

    def check_following(self, value):
        """Checks if the response contains the expected number of words."""
        num_words = processor.count_words(value)

        if self._comparison_relation == COMPARISON_RELATION[0]:  # less than
            return num_words < self._num_words
        elif self._comparison_relation == COMPARISON_RELATION[1]:  # at least
            return num_words >= self._num_words


@instruction_registry.register(_LENGTH + "nth_paragraph_first_word")
class ParagraphFirstWordCheck(BaseInstruction):
    """Check if the first word of a specific paragraph matches a requirement.
    
    Example description:
    "There should be {num_paragraphs} paragraphs.
    Paragraphs and only paragraphs are separated with each other by two
    new lines as if it was '\\n\\n' in python.
    Paragraph {nth_paragraph} must start with word {first_word}."
    """

    def __init__(self, num_paragraphs, nth_paragraph, first_word):
        """Initialize the paragraph first word checker.
        
        Args:
            num_paragraphs: An integer indicating the number of paragraphs expected
                in the response.
            nth_paragraph: An integer indicating the paragraph number to check.
                Note that n starts from 1.
            first_word: A string that represent the first word of the paragraph.
        """
        self._num_paragraphs = num_paragraphs

        if nth_paragraph <= 0 or nth_paragraph > num_paragraphs:
            raise ValueError(f"nth_paragraph must be between 1 and {num_paragraphs}")
        
        self._nth_paragraph = nth_paragraph
        self._first_word = first_word.lower()

    def get_instruction_args(self):
        """Returns the keyword args of the instruction."""
        return {
            "num_paragraphs": self._num_paragraphs,
            "nth_paragraph": self._nth_paragraph,
            "first_word": self._first_word,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of the instruction."""
        return ["num_paragraphs", "nth_paragraph", "first_word"]

    def check_following(self, value):
        """Checks for required number of paragraphs and correct first word.
        
        Args:
            value: a string representing the response.
            
        Returns:
            True if requirements are met, False otherwise.
        """
        paragraphs = re.split(r"\n\n", value)
        num_paragraphs = len(paragraphs)

        for paragraph in paragraphs:
            if not paragraph.strip():
                num_paragraphs -= 1

        # check that index doesn't go out of bounds
        if self._nth_paragraph <= num_paragraphs:
            paragraph = paragraphs[self._nth_paragraph - 1].strip()
            if not paragraph:
                return False
        else:
            return False

        first_word = ""
        punctuation = {".", ",", "?", "!", "'", '"'}

        # get first word and remove punctuation
        word = paragraph.split()[0].strip()
        # Remove leading quotes
        word = word.lstrip("'")
        word = word.lstrip('"')

        for letter in word:
            if letter in punctuation:
                break
            first_word += letter.lower()

        return (
            num_paragraphs == self._num_paragraphs
            and first_word == self._first_word
        )


@instruction_registry.register(_KEYWORD + "forbidden_words")
class ForbiddenWords(BaseInstruction):
    """Check if response doesn't contain specified forbidden words.
    
    Example description:
    "Do not include keywords {forbidden_words} in the response."
    """

    def __init__(self, forbidden_words):
        """Initialize the forbidden words checker.
        
        Args:
            forbidden_words: A sequences of strings representing words that are not
                allowed in the response.
        """
        self._forbidden_words = sorted(list(set(forbidden_words)))

    def get_instruction_args(self):
        """Returns the keyword args of the instruction."""
        return {"forbidden_words": self._forbidden_words}

    def get_instruction_args_keys(self):
        """Returns the args keys of the instruction."""
        return ["forbidden_words"]

    def check_following(self, value):
        """Check if the response does not contain the forbidden words."""
        for word in self._forbidden_words:
            if re.search(r"\b" + word + r"\b", value, flags=re.IGNORECASE):
                return False
        return True


@instruction_registry.register(_CHANGE_CASES + "english_capital")
class CapitalLettersEnglishChecker(BaseInstruction):
    """Check if response is in English with all capital letters.
    
    Example description:
    "Your entire response should be in English, and in all capital letters."
    """

    def __init__(self):
        """Initialize the capital letters checker."""
        pass

    def get_instruction_args(self):
        """Returns the keyword args of the instruction."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of the instruction."""
        return []

    def check_following(self, value):
        """Checks that the response is in English and in all capital letters."""
        assert isinstance(value, str)

        try:
            return value.isupper() and langdetect.detect(value) == "en"
        except langdetect.LangDetectException as e:
            # Count as instruction is followed.
            logging.error(
                "Unable to detect language for text %s due to %s", value, e
            )
            return True


@instruction_registry.register(_CHANGE_CASES + "english_lowercase")
class LowercaseLettersEnglishChecker(BaseInstruction):
    """Check if response is in English with all lowercase letters.
    
    Example description:
    "Your entire response should be in English, and in all lowercase
    letters. No capital letters are allowed."
    """

    def __init__(self):
        """Initialize the lowercase letters checker."""
        pass

    def get_instruction_args(self):
        """Returns the keyword args of the instruction."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of the instruction."""
        return []

    def check_following(self, value):
        """Checks that the response is in English and in all lowercase letters."""
        assert isinstance(value, str)

        try:
            return value.islower() and langdetect.detect(value) == "en"
        except langdetect.LangDetectException as e:
            # Count as instruction is followed.
            logging.error(
                "Unable to detect language for text %s due to %s", value, e
            )
            return True


@instruction_registry.register(_CHANGE_CASES + "capital_word_frequency")
class CapitalWordFrequencyChecker(BaseInstruction):
    """Check frequency of words with all capital letters.
    
    Example description:
    "In your response, words with all capital letters should appear
    {relation} {frequency} times."
    """

    def __init__(self, capital_frequency, capital_relation):
        """Initialize the capital word frequency checker.
        
        Args:
            capital_frequency: An integer that represents the number of words that
                should be in all capital letters.
            capital_relation: A string that is 'at least' or 'at most' that refers to
                the frequency.
        """
        self._frequency = capital_frequency
        
        if capital_relation not in COMPARISON_RELATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                f"{COMPARISON_RELATION}, but {capital_relation} is given."
            )
        self._comparison_relation = capital_relation

    def get_instruction_args(self):
        """Returns the keyword args of the instruction."""
        return {
            "capital_frequency": self._frequency,
            "capital_relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of the instruction."""
        return ["capital_frequency", "capital_relation"]

    def check_following(self, value):
        """Checks the frequency of words with all capital letters."""
        # Use language-specific word tokenizer
        words = processor.word_tokenize(value)
        capital_words = [word for word in words if word.isupper()]

        capital_words_count = len(capital_words)

        if self._comparison_relation == COMPARISON_RELATION[0]:  # less than
            return capital_words_count < self._frequency
        else:  # at least
            return capital_words_count >= self._frequency