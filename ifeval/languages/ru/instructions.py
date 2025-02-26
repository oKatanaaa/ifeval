"""Russian language instruction implementations."""

import collections
import json
import random
import re
import string
from typing import Dict, List, Optional, Sequence, Union, Set, Any

from absl import logging

from ftlangdetect import detect
import nltk

from ifeval.core.instructions import BaseInstruction
from ifeval.core.registry import InstructionRegistry
from ifeval.languages.ru.constants import (
    COMPARISON_RELATION,
    CONSTRAINED_RESPONSE_OPTIONS,
    ENDING_OPTIONS,
    STARTER_OPTIONS,
    POSTSCRIPT_MARKER,
    SECTION_SPLITER,
    NUM_HIGHLIGHTED_SECTIONS,
    NUM_PARAGRAPHS,
    NUM_SECTIONS,
    KEYWORD_FREQUENCY,
    LETTER_FREQUENCY,
    ALL_CAPITAL_WORD_FREQUENCY,
    NUM_WORDS_LOWER_LIMIT,
    NUM_WORDS_UPPER_LIMIT,
    NUM_BULLETS,
    generate_keywords
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
    QuotationChecker
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

    def __init__(self, num_sentences=None, relation=None):
        """Initialize the sentence number checker.
        
        Args:
            num_sentences: An integer specifying the number of sentences as a
                threshold.
            relation: A string in (`less than`, `at least`), defining the relational
                operator for comparison.
        """
        # The number of sentences as a threshold for comparison.
        self._num_sentences_threshold = num_sentences
        if self._num_sentences_threshold is None or self._num_sentences_threshold < 0:
            from ifeval.languages.ru.constants import MAX_NUM_SENTENCES
            self._num_sentences_threshold = random.randint(1, MAX_NUM_SENTENCES)

        if relation is None:
            self._comparison_relation = random.choice(COMPARISON_RELATION)
        elif relation not in COMPARISON_RELATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                f"{COMPARISON_RELATION}, but {relation} is given."
            )
        else:
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


@instruction_registry.register(_FORMAT + "multiple_sections")
class SectionChecker(BaseInstruction):
    """Check if response contains multiple sections.
    
    Example description:
    "Ваш ответ должен содержать {num_sections} разделов. Обозначьте начало
    каждого раздела с помощью {section_spliter} X, например:
    {section_spliter} 1
    [содержание раздела 1]
    {section_spliter} 2
    [содержание раздела 2]"
    """

    def __init__(self, section_spliter=None, num_sections=None):
        """Initialize the section checker.
        
        Args:
            section_spliter: A string represents the section spliter keyword that
                marks a new section, i.e., `Section` or `SECTION`.
            num_sections: An integer specifying the number of sections.
        """
        self._section_spliter = (
            section_spliter.strip()
            if isinstance(section_spliter, str)
            else section_spliter
        )
        if self._section_spliter is None:
            self._section_spliter = random.choice(SECTION_SPLITER)

        self._num_sections = num_sections
        if self._num_sections is None or self._num_sections < 0:
            self._num_sections = random.randint(1, NUM_SECTIONS)

    def get_instruction_args(self):
        """Returns the keyword args of the instruction."""
        return {
            "section_spliter": self._section_spliter,
            "num_sections": self._num_sections,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of the instruction."""
        return ["section_spliter", "num_sections"]

    def check_following(self, value):
        """Checks the response contains multiple sections.
        
        Args:
            value: A string representing the response.
            
        Returns:
            True if the number of sections in the response is sufficient.
        """
        section_splitter_patten = r"\s?" + self._section_spliter + r"\s?(?:[0-9]|[a-zA-Z])"
        sections = re.split(section_splitter_patten, value)
        num_sections = len(sections) - 1
        return num_sections >= self._num_sections


@instruction_registry.register(_CONTENT + "postscript")
class PostscriptChecker(BaseInstruction):
    """Check if response contains a postscript.
    
    Example description:
    "В конце вашего ответа, пожалуйста, добавьте постскриптум,
    начинающийся с {postscript}"
    """

    def __init__(self, postscript_marker=None):
        """Initialize the postscript checker.
        
        Args:
            postscript_marker: A string containing the keyword that marks the start
                of the postscript section.
        """
        self._postscript_marker = (
            postscript_marker.strip()
            if isinstance(postscript_marker, str)
            else postscript_marker
        )
        if self._postscript_marker is None:
            self._postscript_marker = random.choice(POSTSCRIPT_MARKER)

    def get_instruction_args(self):
        """Returns the keyword args of the instruction."""
        return {"postscript_marker": self._postscript_marker}

    def get_instruction_args_keys(self):
        """Returns the args keys of the instruction."""
        return ["postscript_marker"]

    def check_following(self, value):
        """Checks if the response follows the postscript format.
        
        Args:
            value: a string representing the response.
            
        Returns:
            True if the response contains a postscript section.
        """
        value = value.lower()
        if self._postscript_marker == "P.P.S":
            postscript_pattern = r"\s*p\.\s?p\.\s?s.*$"
        elif self._postscript_marker == "P.S.":
            postscript_pattern = r"\s*p\.\s?s\..*$"
        else:
            postscript_pattern = r"\s*" + self._postscript_marker.lower() + r".*$"
        postscript = re.findall(postscript_pattern, value, flags=re.MULTILINE)
        return True if postscript else False


@instruction_registry.register(_KEYWORD + "existence")
class KeywordChecker(BaseInstruction):
    """Check if response contains specified keywords.
    
    Example description:
    "Включите ключевые слова {keywords} в ответ."
    """

    def __init__(self, keywords=None):
        """Initialize the keyword checker.
        
        Args:
            keywords: A sequence of strings representing the keywords that are
                expected in the response.
        """
        if not keywords:
            self._keywords = generate_keywords(num_keywords=2)
        else:
            self._keywords = keywords
        self._keywords = sorted(self._keywords)

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

    def __init__(self, keyword=None, frequency=None, relation=None):
        """Initialize the keyword frequency checker.
        
        Args:
            keyword: A string representing a keyword that is expected in the response.
            frequency: An integer specifying the number of times `keyword` is expected
                to appear in the response.
            relation: A string in (`less than`, `at least`), defining the relational
                operator for comparison.
        """
        if not keyword:
            self._keyword = generate_keywords(num_keywords=1)[0]
        else:
            self._keyword = keyword.strip()

        self._frequency = frequency
        if self._frequency is None or self._frequency < 0:
            self._frequency = random.randint(1, KEYWORD_FREQUENCY)

        if relation is None:
            self._comparison_relation = random.choice(COMPARISON_RELATION)
        elif relation not in COMPARISON_RELATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                f"{COMPARISON_RELATION}, but {relation} is given."
            )
        else:
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

    def __init__(self, num_words=None, relation=None):
        """Initialize the word count checker.
        
        Args:
            num_words: An integer specifying the number of words contained in the
                response.
            relation: A string in (`less than`, `at least`), defining the relational
                operator for comparison.
        """
        self._num_words = num_words
        if self._num_words is None or self._num_words < 0:
            self._num_words = random.randint(
                NUM_WORDS_LOWER_LIMIT, NUM_WORDS_UPPER_LIMIT
            )

        if relation is None:
            self._comparison_relation = random.choice(COMPARISON_RELATION)
        elif relation not in COMPARISON_RELATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                f"{COMPARISON_RELATION}, but {relation} is given."
            )
        else:
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

    def __init__(self, num_paragraphs=None, nth_paragraph=None, first_word=None):
        """Initialize the paragraph first word checker.
        
        Args:
            num_paragraphs: An integer indicating the number of paragraphs expected
                in the response.
            nth_paragraph: An integer indicating the paragraph number to check.
                Note that n starts from 1.
            first_word: A string that represent the first word of the paragraph.
        """
        self._num_paragraphs = num_paragraphs
        if self._num_paragraphs is None or self._num_paragraphs < 0:
            self._num_paragraphs = random.randint(1, NUM_PARAGRAPHS)

        self._nth_paragraph = nth_paragraph
        if (
            self._nth_paragraph is None
            or self._nth_paragraph <= 0
            or self._nth_paragraph > self._num_paragraphs
        ):
            self._nth_paragraph = random.randint(1, self._num_paragraphs + 1)

        self._first_word = first_word
        if self._first_word is None:
            self._first_word = generate_keywords(num_keywords=1)[0]
        self._first_word = self._first_word.lower()

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

    def __init__(self, forbidden_words=None):
        """Initialize the forbidden words checker.
        
        Args:
            forbidden_words: A sequences of strings representing words that are not
                allowed in the response.
        """
        if not forbidden_words:
            self._forbidden_words = generate_keywords(num_keywords=2)
        else:
            self._forbidden_words = list(set(forbidden_words))
        self._forbidden_words = sorted(self._forbidden_words)

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


@instruction_registry.register(_COMBINATION + "repeat_prompt")
class RepeatPromptThenAnswer(BaseInstruction):
    """Check if response begins by repeating the prompt.
    
    Example description:
    "Сначала повторите запрос слово в слово без изменений,
    затем дайте свой ответ (1. не говорите никаких слов или символов
    перед повторением запроса; 2. запрос, который нужно повторить,
    не включает это предложение)"
    """

    def __init__(self, prompt_to_repeat=None):
        """Initialize the repeat prompt checker.
        
        Args:
            prompt_to_repeat: The prompt that is meant to be repeated.
        """
        if not prompt_to_repeat:
            raise ValueError("prompt_to_repeat must be set.")
        else:
            self._prompt_to_repeat = prompt_to_repeat

    def get_instruction_args(self):
        """Returns the keyword args of the instruction."""
        return {"prompt_to_repeat": self._prompt_to_repeat}

    def get_instruction_args_keys(self):
        """Returns the args keys of the instruction."""
        return ["prompt_to_repeat"]

    def check_following(self, value):
        """Check if the response starts by repeating the prompt."""
        if value.strip().lower().startswith(self._prompt_to_repeat.strip().lower()):
            return True
        return False


@instruction_registry.register(_STARTEND + "end_checker")
class EndChecker(BaseInstruction):
    """Check if response ends with a specific phrase.
    
    Example description:
    "Завершите свой ответ точной фразой {ender}.
    Никаких других слов не должно следовать за этой фразой."
    """

    def __init__(self, end_phrase=None):
        """Initialize the end phrase checker.
        
        Args:
            end_phrase: A string representing the phrase the response should end with.
        """
        self._end_phrase = (
            end_phrase.strip() if isinstance(end_phrase, str) else end_phrase
        )
        if self._end_phrase is None:
            self._end_phrase = random.choice(ENDING_OPTIONS)

    def get_instruction_args(self):
        """Returns the keyword args of the instruction."""
        return {"end_phrase": self._end_phrase}

    def get_instruction_args_keys(self):
        """Returns the args keys of the instruction."""
        return ["end_phrase"]

    def check_following(self, value):
        """Checks if the response ends with the expected phrase."""
        value = value.strip().strip('"').lower()
        self._end_phrase = self._end_phrase.strip().lower()
        return value.endswith(self._end_phrase)


@instruction_registry.register(_KEYWORD + "letter_frequency")
class LetterFrequencyChecker(BaseInstruction):
    """Check if a letter appears in the response with specified frequency.
    
    Example description:
    "В вашем ответе буква {letter} должна встречаться {let_relation}
    {let_frequency} раз."
    """

    def __init__(self, letter=None, let_frequency=None, let_relation=None):
        """Initialize the letter frequency checker.
        
        Args:
            letter: A string representing a letter that is expected in the response.
            let_frequency: An integer specifying the number of times the letter is
                expected to appear in the response.
            let_relation: A string in (`less than`, `at least`), defining the
                relational operator for comparison.
        """
        if not letter or len(letter) > 1:
            # For Russian, use the Cyrillic alphabet
            self._letter = random.choice("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
        else:
            self._letter = letter.strip()
        self._letter = self._letter.lower()

        self._frequency = let_frequency
        if self._frequency is None or self._frequency < 0:
            self._frequency = random.randint(1, LETTER_FREQUENCY)

        if let_relation is None:
            self._comparison_relation = random.choice(COMPARISON_RELATION)
        elif let_relation not in COMPARISON_RELATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                f"{COMPARISON_RELATION}, but {let_relation} is given."
            )
        else:
            self._comparison_relation = let_relation

    def get_instruction_args(self):
        """Returns the keyword args of the instruction."""
        return {
            "letter": self._letter,
            "let_frequency": self._frequency,
            "let_relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of the instruction."""
        return ["letter", "let_frequency", "let_relation"]

    def check_following(self, value):
        """Checks that the response contains the letter at the right frequency."""
        value = value.lower()
        letters = collections.Counter(value)

        if self._comparison_relation == "less than":
            return letters[self._letter] < self._frequency
        else:  # at least
            return letters[self._letter] >= self._frequency


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

    def __init__(self, capital_frequency=None, capital_relation=None):
        """Initialize the capital word frequency checker.
        
        Args:
            capital_frequency: An integer that represents the number of words that
                should be in all capital letters.
            capital_relation: A string that is 'at least' or 'at most' that refers to
                the frequency.
        """
        self._frequency = capital_frequency
        if self._frequency is None:
            self._frequency = random.randint(1, ALL_CAPITAL_WORD_FREQUENCY)

        self._comparison_relation = capital_relation
        if capital_relation is None:
            self._comparison_relation = random.choice(COMPARISON_RELATION)
        elif capital_relation not in COMPARISON_RELATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                f"{COMPARISON_RELATION}, but {capital_relation} is given."
            )

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
        # Hyphenated words will count as one word
        words = nltk.word_tokenize(value)
        capital_words = [word for word in words if word.isupper()]

        capital_words_count = len(capital_words)

        if self._comparison_relation == "less than":
            return capital_words_count < self._frequency
        else:  # at least
            return capital_words_count >= self._frequency