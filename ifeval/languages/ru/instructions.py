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

# Create registry and processor instances
instruction_registry = InstructionRegistry()
processor = RussianProcessor()


class ResponseLanguageChecker(BaseInstruction):
    """Check the language of the entire response."""

    def build_description(self, *, language=None):
        """Build the instruction description.

        Args:
            language: A string representing the expected language of the response.

        Returns:
            A string representing the instruction description.
        """
        from ifeval.languages.language_registry import LANGUAGE_CODES
        
        self._language = language
        if self._language is None:
            self._language = random.choice(list(LANGUAGE_CODES.keys()))
        
        self._description_pattern = (
            "Весь ваш ответ должен быть на {language} языке, никакой другой "
            + "язык не допускается."
        )
        return self._description_pattern.format(language=LANGUAGE_CODES[self._language])

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"language": self._language}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
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


class NumberOfSentences(BaseInstruction):
    """Check the number of sentences."""

    def build_description(self, *, num_sentences=None, relation=None):
        """Build the instruction description.

        Args:
            num_sentences: An integer specifying the number of sentences as a
                threshold.
            relation: A string in (`less than`, `at least`), defining the relational
                operator for comparison.

        Returns:
            A string representing the instruction description.
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

        if self._comparison_relation == "less than":
            relation_ru = "менее"
        else:  # "at least"
            relation_ru = "не менее"

        self._description_pattern = (
            "Ваш ответ должен содержать {relation} {num_sentences} предложений."
        )
        return self._description_pattern.format(
            relation=relation_ru,
            num_sentences=self._num_sentences_threshold,
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {
            "num_sentences": self._num_sentences_threshold,
            "relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
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


class PlaceholderChecker(BaseInstruction):
    """Check the placeholders in template writing."""

    def build_description(self, *, num_placeholders=None):
        """Build the instruction description.

        Args:
            num_placeholders: An integer denoting the minimum number of
                placeholders required in the response.

        Returns:
            A string representing the instruction description.
        """
        self._num_placeholders = num_placeholders
        if self._num_placeholders is None or self._num_placeholders < 0:
            from ifeval.languages.ru.constants import NUM_PLACEHOLDERS
            self._num_placeholders = random.randint(1, NUM_PLACEHOLDERS)
            
        self._description_pattern = (
            "Ответ должен содержать не менее {num_placeholders} заполнителей, "
            + "представленных квадратными скобками, например [адрес]."
        )
        return self._description_pattern.format(num_placeholders=self._num_placeholders)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"num_placeholders": self._num_placeholders}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_placeholders"]

    def check_following(self, value):
        """Check if the number of placeholders follows the instruction.

        Args:
            value: A string representing the response.

        Returns:
            True if the actual number of placeholders in the response is greater than
            or equal to `num_placeholders`; otherwise, False.
        """
        placeholders = re.findall(r"\[.*?\]", value)
        num_placeholders = len(placeholders)
        return num_placeholders >= self._num_placeholders


class BulletListChecker(BaseInstruction):
    """Checks the bullet list in the prompt."""

    def build_description(self, *, num_bullets=None):
        """Build the instruction description.

        Args:
            num_bullets: An integer specifying the exact number of bullet lists
                that is required to appear in the response.

        Returns:
            A string representing the instruction description.
        """
        self._num_bullets = num_bullets
        if self._num_bullets is None or self._num_bullets < 0:
            self._num_bullets = random.randint(1, NUM_BULLETS)
            
        self._description_pattern = (
            "Ваш ответ должен содержать ровно {num_bullets} пунктов. "
            + "Используйте маркеры в формате markdown, например:\n"
            + "* Это пункт 1. \n"
            + "* Это пункт 2"
        )
        return self._description_pattern.format(num_bullets=self._num_bullets)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"num_bullets": self._num_bullets}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_bullets"]

    def check_following(self, value):
        r"""Check if the number of bullet lists meets the requirement.

        Args:
            value: A string representing the response. The response is expected to
                contain some bullet lists that start with `\*`.

        Returns:
            True if the actual number of bullet lists in the response meets the
            requirement.
        """
        bullet_lists = re.findall(r"^\s*\*[^\*].*$", value, flags=re.MULTILINE)
        bullet_lists_2 = re.findall(r"^\s*-.*$", value, flags=re.MULTILINE)
        num_bullet_lists = len(bullet_lists) + len(bullet_lists_2)
        return num_bullet_lists == self._num_bullets


class ConstrainedResponseChecker(BaseInstruction):
    """Checks the constrained response."""

    def build_description(self):
        """Build the instruction description."""
        # A sequence of string(s) representing the options of the expected response.
        self._constrained_responses = CONSTRAINED_RESPONSE_OPTIONS
        self._description_pattern = (
            "Ответьте одним из следующих вариантов: {response_options}"
        )
        return self._description_pattern.format(
            response_options=self._constrained_responses
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if the response matches the constrained options.

        Args:
            value: A string representing the response.

        Returns:
            True if the actual response contains one of the options in the constrained
            responses; otherwise False.
        """
        value = value.strip()
        for constrained_response in self._constrained_responses:
            if constrained_response in value:
                return True
        return False


class HighlightSectionChecker(BaseInstruction):
    """Checks the highlighted section."""

    def build_description(self, *, num_highlights=None):
        """Build the instruction description.

        Args:
            num_highlights: An integer specifying the minimum number of highlighted
                sections.

        Returns:
            A string representing the instruction description.
        """
        self._num_highlights = num_highlights
        if self._num_highlights is None or self._num_highlights < 0:
            self._num_highlights = random.randint(1, NUM_HIGHLIGHTED_SECTIONS)

        self._description_pattern = (
            "Выделите не менее {num_highlights} разделов в вашем ответе с помощью "
            + "markdown, например: *выделенный раздел*."
        )

        return self._description_pattern.format(num_highlights=self._num_highlights)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"num_highlights": self._num_highlights}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_highlights"]

    def check_following(self, value):
        """Checks if the number of highlighted sections meets the requirement.

        Args:
            value: a string repesenting the response. The response is expected to
                contain highlighted sections in the format of *highlighted*.

        Returns:
            True if the actual number of highlighted sections in the format of
            *highlighed sections* meets the minimum requirement; otherwise False.
        """
        num_highlights = 0
        highlights = re.findall(r"\*[^\n\*]*\*", value)
        double_highlights = re.findall(r"\*\*[^\n\*]*\*\*", value)
        for highlight in highlights:
            if highlight.strip("*").strip():
                num_highlights += 1
        for highlight in double_highlights:
            if highlight.removeprefix("**").removesuffix("**").strip():
                num_highlights += 1

        return num_highlights >= self._num_highlights


class SectionChecker(BaseInstruction):
    """Checks the sections."""

    def build_description(self, *, section_spliter=None, num_sections=None):
        """Build the instruction description.

        Args:
            section_spliter: A string represents the section spliter keyword that
                marks a new section, i.e., `Section` or `SECTION`.
            num_sections: An integer specifying the number of sections.

        Returns:
            A string representing the instruction description.
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

        self._description_pattern = (
            "Ваш ответ должен содержать {num_sections} разделов. Обозначьте начало "
            + "каждого раздела с помощью {section_spliter} X, например:\n"
            + "{section_spliter} 1\n"
            + "[содержание раздела 1]\n"
            + "{section_spliter} 2\n"
            + "[содержание раздела 2]"
        )

        return self._description_pattern.format(
            num_sections=self._num_sections, section_spliter=self._section_spliter
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {
            "section_spliter": self._section_spliter,
            "num_sections": self._num_sections,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["section_spliter", "num_sections"]

    def check_following(self, value):
        """Checks the response contains multiple sections.

        Args:
            value: A string representing the response. The response is expected
                to contain multiple sections (number of sections is greater than 1).
                A new section starts with `Section 1`, where the number denotes the
                section index.

        Returns:
            True if the number of sections in the response is greater than or equal to
            the minimum number of sections; otherwise, False.
        """
        section_splitter_patten = r"\s?" + self._section_spliter + r"\s?(?:[0-9]|[a-zA-Z])"
        sections = re.split(section_splitter_patten, value)
        num_sections = len(sections) - 1
        return num_sections >= self._num_sections


class ParagraphChecker(BaseInstruction):
    """Checks the paragraphs."""

    def build_description(self, *, num_paragraphs=None):
        """Build the instruction description.

        Args:
            num_paragraphs: An integer specifying the number of paragraphs.

        Returns:
            A string representing the instruction description.
        """
        self._num_paragraphs = num_paragraphs
        if self._num_paragraphs is None or self._num_paragraphs < 0:
            self._num_paragraphs = random.randint(1, NUM_PARAGRAPHS)

        self._description_pattern = (
            "Должно быть {num_paragraphs} абзацев. "
            + "Абзацы разделяются с помощью markdown-разделителя: ***"
        )

        return self._description_pattern.format(num_paragraphs=self._num_paragraphs)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"num_paragraphs": self._num_paragraphs}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_paragraphs"]

    def check_following(self, value):
        """Checks the response contains required number of paragraphs.

        Args:
            value: A string representing the response. The response may contain
                paragraphs that are separated by the markdown divider: `***`.

        Returns:
            True if the actual number of paragraphs is the same as required;
            otherwise, False.
        """
        paragraphs = re.split(r"\s?\*\*\*\s?", value)
        num_paragraphs = len(paragraphs)

        for index, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                if index == 0 or index == len(paragraphs) - 1:
                    num_paragraphs -= 1
                else:
                    return False

        return num_paragraphs == self._num_paragraphs


class PostscriptChecker(BaseInstruction):
    """Checks the postscript."""

    def build_description(self, *, postscript_marker=None):
        """Build the instruction description.

        Args:
            postscript_marker: A string containing the keyword that marks the start
                of the postscript section.

        Returns:
            A string representing the instruction description.
        """
        self._postscript_marker = (
            postscript_marker.strip()
            if isinstance(postscript_marker, str)
            else postscript_marker
        )
        if self._postscript_marker is None:
            self._postscript_marker = random.choice(POSTSCRIPT_MARKER)

        self._description_pattern = (
            "В конце вашего ответа, пожалуйста, добавьте постскриптум, "
            + "начинающийся с {postscript}"
        )

        return self._description_pattern.format(postscript=self._postscript_marker)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"postscript_marker": self._postscript_marker}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["postscript_marker"]

    def check_following(self, value):
        """Checks if the response follows the postscript format.

        Args:
            value: a string representing the response. The response is expected to
                contain a postscript section.

        Returns:
            True if the response contains a postscript section starting with
            the keyword containing in the `instruction_args`; otherwise False.
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


class KeywordChecker(BaseInstruction):
    """Check the existence of certain keywords."""

    def build_description(self, *, keywords=None):
        """Build the instruction description.

        Args:
            keywords: A sequence of strings representing the keywords that are
                expected in the response.

        Returns:
            A string representing the instruction description.
        """

        if not keywords:
            self._keywords = generate_keywords(num_keywords=2)
        else:
            self._keywords = keywords
        self._keywords = sorted(self._keywords)

        self._description_pattern = (
            "Включите ключевые слова {keywords} в ответ."
        )

        return self._description_pattern.format(keywords=self._keywords)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"keywords": self._keywords}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["keywords"]

    def check_following(self, value):
        """Check if the response contains the expected keywords."""
        for keyword in self._keywords:
            if not re.search(processor.lemmatize(keyword), processor.lemmatize(value), flags=re.IGNORECASE):
                return False
        return True


class KeywordFrequencyChecker(BaseInstruction):
    """Check the keyword frequency."""

    def build_description(self, *, keyword=None, frequency=None, relation=None):
        """Build the instruction description.

        Args:
            keyword: A string representing a keyword that is expected in the response.
            frequency: An integer specifying the number of times `keyword` is expected
                to appear in the response.
            relation: A string in (`less than`, `at least`), defining the relational
                operator for comparison.

        Returns:
            A string representing the instruction description.
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

        if self._comparison_relation == "less than":
            relation_ru = "менее"
        else:  # "at least"
            relation_ru = "не менее"

        self._description_pattern = (
            "В вашем ответе слово {keyword} должно встречаться {relation} "
            + "{frequency} раз."
        )

        return self._description_pattern.format(
            keyword=self._keyword,
            relation=relation_ru,
            frequency=self._frequency,
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {
            "keyword": self._keyword,
            "frequency": self._frequency,
            "relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
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


class NumberOfWords(BaseInstruction):
    """Checks the number of words."""

    def build_description(self, *, num_words=None, relation=None):
        """Build the instruction description.

        Args:
            num_words: An integer specifying the number of words contained in the
                response.
            relation: A string in (`less than`, `at least`), defining the relational
                operator for comparison.

        Returns:
            A string representing the instruction description.
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

        if self._comparison_relation == "less than":
            relation_ru = "менее"
        else:  # "at least"
            relation_ru = "не менее"

        self._description_pattern = (
            "Ответьте, используя {relation} {num_words} слов."
        )

        return self._description_pattern.format(
            relation=relation_ru, num_words=self._num_words
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"num_words": self._num_words, "relation": self._comparison_relation}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_words", "relation"]

    def check_following(self, value):
        """Checks if the response contains the expected number of words."""
        num_words = processor.count_words(value)

        if self._comparison_relation == "less than":
            return num_words < self._num_words
        elif self._comparison_relation == "at least":
            return num_words >= self._num_words


class JsonFormat(BaseInstruction):
    """Check the Json format."""

    def build_description(self):
        self._description_pattern = (
            "Весь вывод должен быть в формате JSON. Вы можете использовать"
            " маркеры markdown, например ```."
        )
        return self._description_pattern

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        value = (
            value.strip()
            .removeprefix("```json")
            .removeprefix("```Json")
            .removeprefix("```JSON")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        try:
            json.loads(value)
        except ValueError as _:
            return False
        return True


class ParagraphFirstWordCheck(BaseInstruction):
    """Check the paragraph and the first word of the nth paragraph."""

    def build_description(
        self, num_paragraphs=None, nth_paragraph=None, first_word=None
    ):
        r"""Build the instruction description.

        Args:
            num_paragraphs: An integer indicating the number of paragraphs expected
                in the response. A paragraph is a subset of the string that is
                expected to be separated by '\n\n'.
            nth_paragraph: An integer indicating the paragraph number that we look at.
                Note that n starts from 1.
            first_word: A string that represents the first word of the bth paragraph.

        Returns:
            A string representing the instruction description.
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

        self._description_pattern = (
            "Должно быть {num_paragraphs} абзацев. "
            + "Абзацы и только абзацы разделяются друг от друга двумя "
            + "переносами строки, как если бы это было '\\n\\n' в python. "
            + "Абзац {nth_paragraph} должен начинаться со слова {first_word}."
        )

        return self._description_pattern.format(
            num_paragraphs=self._num_paragraphs,
            nth_paragraph=self._nth_paragraph,
            first_word=self._first_word,
        )

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {
            "num_paragraphs": self._num_paragraphs,
            "nth_paragraph": self._nth_paragraph,
            "first_word": self._first_word,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_paragraphs", "nth_paragraph", "first_word"]

    def check_following(self, value):
        """Checks for required number of paragraphs and correct first word.

        Args:
            value: a string representing the response. The response may contain
                paragraphs that are separated by two new lines and the first word of
                the nth paragraph will have to match a specified word.

        Returns:
            True if the number of paragraphs is the same as required and the first
            word of the specified paragraph is the same as required. Otherwise, false.
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


class ForbiddenWords(BaseInstruction):
    """Checks that specified words are not used in response."""

    def build_description(self, forbidden_words=None):
        """Build the instruction description.

        Args:
            forbidden_words: A sequences of strings representing words that are not
                allowed in the response.

        Returns:
            A string representing the instruction description.
        """
        if not forbidden_words:
            self._forbidden_words = generate_keywords(num_keywords=2)
        else:
            self._forbidden_words = list(set(forbidden_words))
        self._forbidden_words = sorted(self._forbidden_words)
        self._description_pattern = (
            "Не включайте ключевые слова {forbidden_words} в ответ."
        )

        return self._description_pattern.format(forbidden_words=self._forbidden_words)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"forbidden_words": self._forbidden_words}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["forbidden_words"]

    def check_following(self, value):
        """Check if the response does not contain the expected keywords."""
        for word in self._forbidden_words:
            if re.search(r"\b" + processor.lemmatize(word) + r"\b", processor.lemmatize(value), flags=re.IGNORECASE):
                return False
        return True


class TwoResponsesChecker(BaseInstruction):
    """Check that two responses were given."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "Дайте два разных ответа. Ответы и только ответы должны быть"
            " разделены 6 символами звездочки: ******."
        )
        return self._description_pattern

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if the response has two different answers.

        Args:
            value: A string representing the response.

        Returns:
            True if two responses are detected and false otherwise.
        """
        valid_responses = list()
        responses = value.split("******")
        for index, response in enumerate(responses):
            if not response.strip():
                if index != 0 and index != len(responses) - 1:
                    return False
            else:
                valid_responses.append(response)
        return (
            len(valid_responses) == 2
            and valid_responses[0].strip() != valid_responses[1].strip()
        )


class RepeatPromptThenAnswer(BaseInstruction):
    """Checks that Prompt is first repeated then answered."""

    def build_description(self, *, prompt_to_repeat=None):
        """Build the instruction description.

        Args:
            prompt_to_repeat: The prompt that is meant to be repeated.

        Returns:
            A string representing the instruction description.
        """
        if not prompt_to_repeat:
            raise ValueError("prompt_to_repeat must be set.")
        else:
            self._prompt_to_repeat = prompt_to_repeat
        self._description_pattern = (
            "Сначала повторите запрос слово в слово без изменений,"
            " затем дайте свой ответ (1. не говорите никаких слов или символов"
            " перед повторением запроса; 2. запрос, который нужно повторить,"
            " не включает это предложение)"
        )
        return self._description_pattern

    def get_instruction_args(self):
        return {"prompt_to_repeat": self._prompt_to_repeat}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["prompt_to_repeat"]

    def check_following(self, value):
        if value.strip().lower().startswith(self._prompt_to_repeat.strip().lower()):
            return True
        return False


class EndChecker(BaseInstruction):
    """Checks that the prompt ends with a given phrase."""

    def build_description(self, *, end_phrase=None):
        """Build the instruction description.

        Args:
            end_phrase: A string representing the phrase the response should end with.

        Returns:
            A string representing the instruction description.
        """
        self._end_phrase = (
            end_phrase.strip() if isinstance(end_phrase, str) else end_phrase
        )
        if self._end_phrase is None:
            self._end_phrase = random.choice(ENDING_OPTIONS)
        self._description_pattern = (
            "Завершите свой ответ точной фразой {ender}. "
            "Никаких других слов не должно следовать за этой фразой."
        )
        return self._description_pattern.format(ender=self._end_phrase)

    def get_instruction_args(self):
        return {"end_phrase": self._end_phrase}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["end_phrase"]

    def check_following(self, value):
        """Checks if the response ends with the expected phrase."""
        value = value.strip().strip('"').lower()
        self._end_phrase = self._end_phrase.strip().lower()
        return value.endswith(self._end_phrase)


class TitleChecker(BaseInstruction):
    """Checks the response for a title."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "Ваш ответ должен содержать заголовок, заключенный в двойные"
            " угловые скобки, например <<стихотворение о радости>>."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if the response contains a title."""
        pattern = r"<<[^\n]+>>"
        re_pattern = re.compile(pattern)
        titles = re.findall(re_pattern, value)

        for title in titles:
            if title.lstrip("<").rstrip(">").strip():
                return True
        return False


class LetterFrequencyChecker(BaseInstruction):
    """Checks letter frequency."""

    def build_description(self, *, letter=None, let_frequency=None, let_relation=None):
        """Build the instruction description.

        Args:
            letter: A string representing a letter that is expected in the response.
            let_frequency: An integer specifying the number of times `keyword` is
                expected to appear in the response.
            let_relation: A string in (`less than`, `at least`), defining the
                relational operator for comparison.

        Returns:
            A string representing the instruction description.
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

        if self._comparison_relation == "less than":
            relation_ru = "менее"
        else:  # "at least"
            relation_ru = "не менее"

        self._description_pattern = (
            "В вашем ответе буква {letter} должна встречаться {let_relation}"
            " {let_frequency} раз."
        )

        return self._description_pattern.format(
            letter=self._letter,
            let_frequency=self._frequency,
            let_relation=relation_ru,
        )

    def get_instruction_args(self):
        """Returns the keyword args of build description."""
        return {
            "letter": self._letter,
            "let_frequency": self._frequency,
            "let_relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["letter", "let_frequency", "let_relation"]

    def check_following(self, value):
        """Checks that the response contains the letter at the right frequency."""
        value = value.lower()
        letters = collections.Counter(value)

        if self._comparison_relation == "less than":
            return letters[self._letter] < self._frequency
        else:  # "at least"
            return letters[self._letter] >= self._frequency


class CapitalLettersRussianChecker(BaseInstruction):
    """Checks that the response is in Russian and is in all capital letters."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "Весь ваш ответ должен быть на русском языке и заглавными буквами."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
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


class LowercaseLettersRussianChecker(BaseInstruction):
    """Checks that the response is in Russian and is in all lowercase letters."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "Весь ваш ответ должен быть на русском языке и строчными"
            " буквами. Заглавные буквы не допускаются."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
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


class CommaChecker(BaseInstruction):
    """Checks the response for no commas."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "В вашем ответе воздержитесь от использования запятых."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks that the response does not contain commas."""
        return not re.search(r",", value)


class CapitalWordFrequencyChecker(BaseInstruction):
    """Checks frequency of words with all capital letters."""

    def build_description(
        self,
        capital_frequency=None,
        capital_relation=None,
    ):
        """Build the instruction description.

        Args:
            capital_frequency: An integer that represents the number of words that
                should be in all capital letters.
            capital_relation: A string that is 'at least' or 'at most' that refers to
                the frequency.

        Returns:
            A string representing the instruction description.
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

        if self._comparison_relation == "less than":
            relation_ru = "менее"
        else:  # "at least"
            relation_ru = "не менее"

        self._description_pattern = (
            "В вашем ответе слова, написанные заглавными буквами, должны"
            " встречаться {relation} {frequency} раз."
        )

        return self._description_pattern.format(
            frequency=self._frequency, relation=relation_ru
        )

    def get_instruction_args(self):
        """Returns the keyword args of build description."""
        return {
            "capital_frequency": self._frequency,
            "capital_relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["capital_frequency", "capital_relation"]

    def check_following(self, value):
        """Checks the frequency of words with all capital letters."""
        # Hyphenated words will count as one word
        words = nltk.word_tokenize(value)
        capital_words = [word for word in words if word.isupper()]

        capital_words_count = len(capital_words)

        if self._comparison_relation == "less than":
            return capital_words_count < self._frequency
        else:  # "at least"
            return capital_words_count >= self._frequency


class QuotationChecker(BaseInstruction):
    """Checks response is wrapped with double quotation marks."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "Заключите весь ваш ответ в двойные кавычки."
        )
        return self._description_pattern

    def get_instruction_args(self):
        """Returns the keyword args of build description."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if the response is wrapped with double quotation marks."""
        value = value.strip()
        return len(value) > 1 and value[0] == '"' and value[-1] == '"'


# Register all instruction classes
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

instruction_registry.register(_KEYWORD + "existence")(KeywordChecker)
instruction_registry.register(_KEYWORD + "frequency")(KeywordFrequencyChecker)
instruction_registry.register(_KEYWORD + "forbidden_words")(ForbiddenWords)
instruction_registry.register(_KEYWORD + "letter_frequency")(LetterFrequencyChecker)
instruction_registry.register(_LANGUAGE + "response_language")(ResponseLanguageChecker)
instruction_registry.register(_LENGTH + "number_sentences")(NumberOfSentences)
instruction_registry.register(_LENGTH + "number_paragraphs")(ParagraphChecker)
instruction_registry.register(_LENGTH + "number_words")(NumberOfWords)
instruction_registry.register(_LENGTH + "nth_paragraph_first_word")(ParagraphFirstWordCheck)
instruction_registry.register(_CONTENT + "number_placeholders")(PlaceholderChecker)
instruction_registry.register(_CONTENT + "postscript")(PostscriptChecker)
instruction_registry.register(_FORMAT + "number_bullet_lists")(BulletListChecker)
instruction_registry.register(_FORMAT + "constrained_response")(ConstrainedResponseChecker)
instruction_registry.register(_FORMAT + "number_highlighted_sections")(HighlightSectionChecker)
instruction_registry.register(_FORMAT + "multiple_sections")(SectionChecker)
instruction_registry.register(_FORMAT + "json_format")(JsonFormat)
instruction_registry.register(_FORMAT + "title")(TitleChecker)
instruction_registry.register(_COMBINATION + "two_responses")(TwoResponsesChecker)
instruction_registry.register(_COMBINATION + "repeat_prompt")(RepeatPromptThenAnswer)
instruction_registry.register(_STARTEND + "end_checker")(EndChecker)
instruction_registry.register(_CHANGE_CASES + "capital_word_frequency")(CapitalWordFrequencyChecker)
instruction_registry.register(_CHANGE_CASES + "english_capital")(CapitalLettersRussianChecker)
instruction_registry.register(_CHANGE_CASES + "english_lowercase")(LowercaseLettersRussianChecker)
instruction_registry.register(_PUNCTUATION + "no_comma")(CommaChecker)
instruction_registry.register(_STARTEND + "quotation")(QuotationChecker)

# Define instruction conflicts
INSTRUCTION_CONFLICTS = {
    _KEYWORD + "existence": {_KEYWORD + "existence"},
    _KEYWORD + "frequency": {_KEYWORD + "frequency"},
    _KEYWORD + "forbidden_words": {_KEYWORD + "forbidden_words"},
    _KEYWORD + "letter_frequency": {_KEYWORD + "letter_frequency"},
    _LANGUAGE
    + "response_language": {
        _LANGUAGE + "response_language",
        _FORMAT + "multiple_sections",
        _KEYWORD + "existence",
        _KEYWORD + "frequency",
        _KEYWORD + "forbidden_words",
        _STARTEND + "end_checker",
        _CHANGE_CASES + "english_capital",
        _CHANGE_CASES + "english_lowercase",
    },
    _LENGTH + "number_sentences": {_LENGTH + "number_sentences"},
    _LENGTH
    + "number_paragraphs": {
        _LENGTH + "number_paragraphs",
        _LENGTH + "nth_paragraph_first_word",
        _LENGTH + "number_sentences",
        _LENGTH + "nth_paragraph_first_word",
    },
    _LENGTH + "number_words": {_LENGTH + "number_words"},
    _LENGTH
    + "nth_paragraph_first_word": {
        _LENGTH + "nth_paragraph_first_word",
        _LENGTH + "number_paragraphs",
    },
    _CONTENT + "number_placeholders": {_CONTENT + "number_placeholders"},
    _CONTENT + "postscript": {_CONTENT + "postscript"},
    _FORMAT + "number_bullet_lists": {_FORMAT + "number_bullet_lists"},
    _FORMAT + "constrained_response": set(instruction_registry._instructions.keys()),
    _FORMAT
    + "number_highlighted_sections": {_FORMAT + "number_highlighted_sections"},
    _FORMAT
    + "multiple_sections": {
        _FORMAT + "multiple_sections",
        _LANGUAGE + "response_language",
        _FORMAT + "number_highlighted_sections",
    },
    _FORMAT
    + "json_format": set(instruction_registry._instructions.keys()).difference(
        {_KEYWORD + "forbidden_words", _KEYWORD + "existence"}
    ),
    _FORMAT + "title": {_FORMAT + "title"},
    _COMBINATION
    + "two_responses": set(instruction_registry._instructions.keys()).difference({
        _KEYWORD + "forbidden_words",
        _KEYWORD + "existence",
        _LANGUAGE + "response_language",
        _FORMAT + "title",
        _PUNCTUATION + "no_comma"
    }),
    _COMBINATION 
    + "repeat_prompt": set(instruction_registry._instructions.keys()).difference({
        _KEYWORD + "existence",
        _FORMAT + "title",
        _PUNCTUATION + "no_comma"
    }),
    _STARTEND + "end_checker": {_STARTEND + "end_checker"},
    _CHANGE_CASES 
    + "capital_word_frequency": {
        _CHANGE_CASES + "capital_word_frequency",
        _CHANGE_CASES + "english_lowercase",
        _CHANGE_CASES + "english_capital",
    },
    _CHANGE_CASES + "english_capital": {_CHANGE_CASES + "english_capital"},
    _CHANGE_CASES 
    + "english_lowercase": {
        _CHANGE_CASES + "english_lowercase",
        _CHANGE_CASES + "english_capital",
    },
    _PUNCTUATION + "no_comma": {_PUNCTUATION + "no_comma"},
    _STARTEND + "quotation": {_STARTEND + "quotation", _FORMAT + "title"},
}

# Make conflicts bidirectional
instruction_registry.conflict_make()