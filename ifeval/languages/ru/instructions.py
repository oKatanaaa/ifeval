"""Russian language instruction implementations."""

import collections
import random
import re

from absl import logging

from ftlangdetect import detect

from ifeval.core.instructions import BaseInstruction
from ifeval.core.registry import InstructionRegistry
from ifeval.languages.ru.constants import (
    COMPARISON_RELATION,
    CONSTRAINED_RESPONSE_OPTIONS
)
from ifeval.languages.ru.processor import RussianProcessor
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
    LetterFrequencyChecker
)

# Create registry and processor instances
instruction_registry = InstructionRegistry()
processor = RussianProcessor()

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

# Language-specific instructions

@instruction_registry.register(_LANGUAGE + "response_language")
class ResponseLanguageChecker(BaseInstruction):
    """Check if response is in a specific language.
    
    Example description:
    "Весь ваш ответ должен быть на {language} языке, никакой другой 
    язык не допускается."
    """

    def __init__(self, language=None):
        """Initialize the language checker.
        
        Args:
            language: A string representing the expected language of the response.
        """
        from ifeval.languages.language_registry import LANGUAGE_CODES
        
        self._language = language
        if self._language is None:
            self._language = random.choice(list(LANGUAGE_CODES.keys()))

    def get_instruction_args(self):
        """Returns the keyword args of the instruction."""
        return {"language": self._language}

    def get_instruction_args_keys(self):
        """Returns the args keys of the instruction."""
        return ["language"]

    def check_following(self, value):
        """Check if the language of the entire response follows the instruction.
        
        Args:
            value: A string representing the response.
            
        Returns:
            True if the language of `value` follows instruction; otherwise False.
        """
        assert isinstance(value, str)

        try:
            return detect(text=value.replace("\n", " "), low_memory=False)["lang"] == self._language
        except Exception as e:
            # Count as instruction is followed.
            logging.error(
                "Unable to detect language for text %s due to %s", value, e
            )
            return True


@instruction_registry.register(_LENGTH + "number_sentences")
class NumberOfSentences(BaseInstruction):
    """Check if response contains specific number of sentences.
    
    Example description:
    "Ваш ответ должен содержать {relation} {num_sentences} предложений."
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
        if self._comparison_relation == "less than":
            return num_sentences < self._num_sentences_threshold
        elif self._comparison_relation == "at least":
            return num_sentences >= self._num_sentences_threshold


@instruction_registry.register(_FORMAT + "constrained_response")
class ConstrainedResponseChecker(BaseInstruction):
    """Check if response contains one of the constrained options.
    
    Example description:
    "Ответьте одним из следующих вариантов: {response_options}"
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
    "Включите ключевые слова {keywords} в ответ."
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
            if not re.search(processor.lemmatize(keyword), processor.lemmatize(value), flags=re.IGNORECASE):
                return False
        return True


@instruction_registry.register(_KEYWORD + "frequency")
class KeywordFrequencyChecker(BaseInstruction):
    """Check if a keyword appears in the response with specified frequency.
    
    Example description:
    "В вашем ответе слово {keyword} должно встречаться {relation}
    {frequency} раз."
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
            re.findall(processor.lemmatize(self._keyword), processor.lemmatize(value), flags=re.IGNORECASE)
        )

        if self._comparison_relation == "less than":
            return actual_occurrences < self._frequency
        elif self._comparison_relation == "at least":
            return actual_occurrences >= self._frequency


@instruction_registry.register(_LENGTH + "number_words")
class NumberOfWords(BaseInstruction):
    """Check if response contains specified number of words.
    
    Example description:
    "Ответьте, используя {relation} {num_words} слов."
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

        if self._comparison_relation == "less than":
            return num_words < self._num_words
        elif self._comparison_relation == "at least":
            return num_words >= self._num_words


@instruction_registry.register(_LENGTH + "nth_paragraph_first_word")
class ParagraphFirstWordCheck(BaseInstruction):
    """Check if the first word of a specific paragraph matches a requirement.
    
    Example description:
    "Должно быть {num_paragraphs} абзацев.
    Абзацы и только абзацы разделяются друг от друга двумя
    переносами строки, как если бы это было '\\n\\n' в python.
    Абзац {nth_paragraph} должен начинаться со слова {first_word}."
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
    "Не включайте ключевые слова {forbidden_words} в ответ."
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
            if re.search(r"\b" + processor.lemmatize(word) + r"\b", processor.lemmatize(value), flags=re.IGNORECASE):
                return False
        return True


@instruction_registry.register(_CHANGE_CASES + "english_capital")
class CapitalLettersRussianChecker(BaseInstruction):
    """Check if response is in Russian with all capital letters.
    
    Example description:
    "Весь ваш ответ должен быть на русском языке и заглавными буквами."
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
        """Checks that the response is in Russian and in all capital letters."""
        assert isinstance(value, str)

        try:
            return value.isupper() and detect(text=value.replace("\n", " "), low_memory=False)["lang"] == "ru"
        except Exception as e:
            # Count as instruction is followed.
            logging.error(
                "Unable to detect language for text %s due to %s", value, e
            )
            return True


@instruction_registry.register(_CHANGE_CASES + "english_lowercase")
class LowercaseLettersRussianChecker(BaseInstruction):
    """Check if response is in Russian with all lowercase letters.
    
    Example description:
    "Весь ваш ответ должен быть на русском языке и строчными
    буквами. Заглавные буквы не допускаются."
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
        """Checks that the response is in Russian and in all lowercase letters."""
        assert isinstance(value, str)

        try:
            return value.islower() and detect(text=value.replace("\n", " "), low_memory=False)["lang"] == "ru"
        except Exception as e:
            # Count as instruction is followed.
            logging.error(
                "Unable to detect language for text %s due to %s", value, e
            )
            return True


@instruction_registry.register(_CHANGE_CASES + "capital_word_frequency")
class CapitalWordFrequencyChecker(BaseInstruction):
    """Check frequency of words with all capital letters.
    
    Example description:
    "В вашем ответе слова, написанные заглавными буквами, должны
    встречаться {relation} {frequency} раз."
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

        if self._comparison_relation == "less than":
            return capital_words_count < self._frequency
        else:  # at least
            return capital_words_count >= self._frequency