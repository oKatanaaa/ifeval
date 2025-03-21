"""Generic language-agnostic instruction implementations."""

import json
from absl import logging
import re
import collections

from ifeval.core import legacy_behavior
from ifeval.core.instructions import BaseInstruction
from ifeval.utils.text_processing import language_is_supported, detect_language

# Default values for instruction parameters
DEFAULT_NUM_PLACEHOLDERS = 4
DEFAULT_NUM_BULLETS = 5
DEFAULT_NUM_HIGHLIGHTS = 4
DEFAULT_NUM_PARAGRAPHS = 5

# Generic constants for use in multilingual code
COMPARISON_RELATION = ("less than", "at least")

class PlaceholderChecker(BaseInstruction):
    """Check if response contains placeholders in square brackets.
    
    Example description:
    "The response must contain at least {num_placeholders} placeholders 
    represented by square brackets, such as [address]."
    """
    
    def __init__(self, num_placeholders):
        """Initialize the placeholder checker.
        
        Args:
            num_placeholders: An integer denoting the minimum number of
                placeholders required in the response.
        """
        self._num_placeholders = num_placeholders
    
    def get_instruction_args(self):
        """Returns the keyword args of the instruction."""
        return {"num_placeholders": self._num_placeholders}
    
    def get_instruction_args_keys(self):
        """Returns the args keys of the instruction."""
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
    """Check if response contains the specified number of bullet points.
    
    Example description:
    "Your answer must contain exactly {num_bullets} bullet points.
    Use the markdown bullet points such as:
    * This is point 1.
    * This is point 2"
    """
    
    def __init__(self, num_bullets):
        """Initialize the bullet list checker.
        
        Args:
            num_bullets: An integer specifying the exact number of bullet lists
                that is required to appear in the response.
        """
        self._num_bullets = num_bullets
    
    def get_instruction_args(self):
        """Returns the keyword args of the instruction."""
        return {"num_bullets": self._num_bullets}
    
    def get_instruction_args_keys(self):
        """Returns the args keys of the instruction."""
        return ["num_bullets"]
    
    def check_following(self, value):
        r"""Check if the number of bullet lists meets the requirement.
        
        Args:
            value: A string representing the response.
            
        Returns:
            True if the actual number of bullet lists in the response meets the
            requirement.
        """
        bullet_lists = re.findall(r"^\s*\*[^\*].*$", value, flags=re.MULTILINE)
        bullet_lists_2 = re.findall(r"^\s*-.*$", value, flags=re.MULTILINE)
        num_bullet_lists = len(bullet_lists) + len(bullet_lists_2)
        return num_bullet_lists == self._num_bullets


class HighlightSectionChecker(BaseInstruction):
    """Check if response contains highlighted sections using markdown.
    
    Example description:
    "Highlight at least {num_highlights} sections in your answer with 
    markdown, i.e. *highlighted section*."
    """
    
    def __init__(self, num_highlights):
        """Initialize the highlighted section checker.
        
        Args:
            num_highlights: An integer specifying the minimum number of highlighted
                sections.
        """
        self._num_highlights = num_highlights
    
    def get_instruction_args(self):
        """Returns the keyword args of the instruction."""
        return {"num_highlights": self._num_highlights}
    
    def get_instruction_args_keys(self):
        """Returns the args keys of the instruction."""
        return ["num_highlights"]
    
    def check_following(self, value):
        """Checks if the number of highlighted sections meets the requirement.
        
        Args:
            value: A string representing the response.
            
        Returns:
            True if the actual number of highlighted sections meets the minimum requirement.
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


class ParagraphChecker(BaseInstruction):
    """Check if response contains the specified number of paragraphs.
    
    Example description:
    "There should be {num_paragraphs} paragraphs.
    Paragraphs are separated with the markdown divider: ***"
    """
    
    def __init__(self, num_paragraphs):
        """Initialize the paragraph checker.
        
        Args:
            num_paragraphs: An integer specifying the number of paragraphs.
        """
        self._num_paragraphs = num_paragraphs
    
    def get_instruction_args(self):
        """Returns the keyword args of the instruction."""
        return {"num_paragraphs": self._num_paragraphs}
    
    def get_instruction_args_keys(self):
        """Returns the args keys of the instruction."""
        return ["num_paragraphs"]
    
    def check_following(self, value):
        """Checks if the response contains required number of paragraphs.
        
        Args:
            value: A string representing the response.
            
        Returns:
            True if the actual number of paragraphs is the same as required.
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


class JsonFormat(BaseInstruction):
    """Check if response is in JSON format.
    
    Example description:
    "Entire output should be wrapped in JSON format. You can use markdown
    ticks such as ```."
    """
    
    def __init__(self):
        """Initialize the JSON format checker."""
        pass
    
    def get_instruction_args(self):
        """Returns the keyword args of the instruction."""
        return None
    
    def get_instruction_args_keys(self):
        """Returns the args keys of the instruction."""
        return []
    
    def check_following(self, value):
        """Check if the response is in valid JSON format.
        
        Args:
            value: A string representing the response.
            
        Returns:
            True if the response is valid JSON.
        """
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
        except ValueError:
            return False
        return True


class TwoResponsesChecker(BaseInstruction):
    """Check if response contains two different responses separated by asterisks.
    
    Example description:
    "Give two different responses. Responses and only responses should
    be separated by 6 asterisk symbols: ******."
    """
    
    def __init__(self):
        """Initialize the two responses checker."""
        pass
    
    def get_instruction_args(self):
        """Returns the keyword args of the instruction."""
        return None
    
    def get_instruction_args_keys(self):
        """Returns the args keys of the instruction."""
        return []
    
    def check_following(self, value):
        """Check if the response has two different answers.
        
        Args:
            value: A string representing the response.
            
        Returns:
            True if two different responses are detected.
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


class TitleChecker(BaseInstruction):
    """Check if response contains a title wrapped in double angular brackets.
    
    Example description:
    "Your answer must contain a title, wrapped in double angular brackets,
    such as <<poem of joy>>."
    """
    
    def __init__(self):
        """Initialize the title checker."""
        pass
    
    def get_instruction_args(self):
        """Returns the keyword args of the instruction."""
        return None
    
    def get_instruction_args_keys(self):
        """Returns the args keys of the instruction."""
        return []
    
    def check_following(self, value):
        """Check if the response contains a title.
        
        Args:
            value: A string representing the response.
            
        Returns:
            True if the response contains a title in double angular brackets.
        """
        pattern = r"<<[^\n]+>>"
        re_pattern = re.compile(pattern)
        titles = re.findall(re_pattern, value)

        for title in titles:
            if title.lstrip("<").rstrip(">").strip():
                return True
        return False


class CommaChecker(BaseInstruction):
    """Check if response does not contain any commas.
    
    Example description:
    "In your entire response, refrain from the use of any commas."
    """
    
    def __init__(self):
        """Initialize the comma checker."""
        pass
    
    def get_instruction_args(self):
        """Returns the keyword args of the instruction."""
        return None
    
    def get_instruction_args_keys(self):
        """Returns the args keys of the instruction."""
        return []
    
    def check_following(self, value):
        """Check if the response does not contain commas.
        
        Args:
            value: A string representing the response.
            
        Returns:
            True if the response does not contain commas.
        """
        return not re.search(r",", value)


class QuotationChecker(BaseInstruction):
    """Check if response is wrapped in double quotation marks.
    
    Example description:
    "Wrap your entire response with double quotation marks."
    """
    
    def __init__(self):
        """Initialize the quotation checker."""
        pass
    
    def get_instruction_args(self):
        """Returns the keyword args of the instruction."""
        return None
    
    def get_instruction_args_keys(self):
        """Returns the args keys of the instruction."""
        return []
    
    def check_following(self, value):
        """Check if the response is wrapped with double quotation marks.
        
        Args:
            value: A string representing the response.
            
        Returns:
            True if the response is wrapped with double quotation marks.
        """
        value = value.strip()
        return len(value) > 1 and value[0] == '"' and value[-1] == '"'


class SectionChecker(BaseInstruction):
    """Check if response contains multiple sections.
    
    Example description:
    "Your response must have {num_sections} sections. Mark the beginning
    of each section with {section_spliter} X, such as:
    {section_spliter} 1
    [content of section 1]
    {section_spliter} 2
    [content of section 2]"
    """

    def __init__(self, section_spliter, num_sections):
        """Initialize the section checker.
        
        Args:
            section_spliter: A string represents the section spliter keyword that
                marks a new section, i.e., `Section` or `SECTION`.
            num_sections: An integer specifying the number of sections.
            allow_letters: Whether to allow letters (in addition to digits) as section numbers.
                           Russian implementation uses True, English uses False.
        """
        self._section_spliter = section_spliter.strip() if isinstance(section_spliter, str) else section_spliter
        self._num_sections = num_sections

    def get_instruction_args(self):
        """Returns the keyword args of the instruction."""
        return {
            "section_spliter": self._section_spliter,
            "num_sections": self._num_sections
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of the instruction."""
        return ["section_spliter", "num_sections", "allow_letters"]

    def check_following(self, value):
        """Checks the response contains multiple sections.
        
        Args:
            value: A string representing the response.
            
        Returns:
            True if the number of sections in the response is sufficient.
        """
        # This is a more general regex compared to original implementation
        # in that it allows for letters
        section_splitter_pattern = r"\s?" + self._section_spliter + r"\s?(?:[0-9]|[a-zA-Z])"
            
        sections = re.split(section_splitter_pattern, value)
        num_sections = len(sections) - 1
        return num_sections >= self._num_sections


class PostscriptChecker(BaseInstruction):
    """Check if response contains a postscript.
    
    Example description:
    "At the end of your response, please explicitly add a postscript
    starting with {postscript}"
    """

    def __init__(self, postscript_marker):
        """Initialize the postscript checker.
        
        Args:
            postscript_marker: A string containing the keyword that marks the start
                of the postscript section.
        """
        self._postscript_marker = postscript_marker.strip() if isinstance(postscript_marker, str) else postscript_marker

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


class RepeatPromptThenAnswer(BaseInstruction):
    """Check if response begins by repeating the prompt.
    
    Example description:
    "First repeat the request word for word without change,
    then give your answer (1. do not say any words or characters
    before repeating the request; 2. the request you need to repeat
    does not include this sentence)"
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


class EndChecker(BaseInstruction):
    """Check if response ends with a specific phrase.
    
    Example description:
    "Finish your response with this exact phrase {ender}.
    No other words should follow this phrase."
    """

    def __init__(self, end_phrase):
        """Initialize the end phrase checker.
        
        Args:
            end_phrase: A string representing the phrase the response should end with.
        """
        self._end_phrase = end_phrase.strip() if isinstance(end_phrase, str) else end_phrase

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


class LetterFrequencyChecker(BaseInstruction):
    """Check if a letter appears in the response with specified frequency.
    
    Example description:
    "In your response, the letter {letter} should appear {let_relation}
    {let_frequency} times."
    """

    def __init__(self, letter, let_frequency, let_relation):
        """Initialize the letter frequency checker.
        
        Args:
            letter: A string representing a letter that is expected in the response.
            let_frequency: An integer specifying the number of times the letter is
                expected to appear in the response.
            let_relation: A string in (`less than`, `at least`), defining the
                relational operator for comparison.
        """
        self._letter = letter.strip().lower()
        self._frequency = let_frequency

        if let_relation not in COMPARISON_RELATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                f"{COMPARISON_RELATION}, but {let_relation} is given."
            )
        
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

        if self._comparison_relation == COMPARISON_RELATION[0]:  # less than
            return letters[self._letter] < self._frequency
        else:  # at least
            return letters[self._letter] >= self._frequency
        

class ResponseLanguageChecker(BaseInstruction):
    """Check if response is in a specific language.
    
    Example description:
    "Your ENTIRE response should be in {language} language, no other 
    language is allowed."
    """

    def __init__(self, language=None):
        """Initialize the language checker.
        
        Args:
            language: A string representing the expected language of the response.
        """
        assert language_is_supported(language), f'Language {language} is not supported.'
        self._language = language

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
            return detect_language(text=value.replace("\n", " ")) == self._language
        except Exception as e:
            logging.error(
                "Unable to detect language for text %s due to %s", value, e
            )
            if legacy_behavior():
                # Count as instruction is followed.
                return True
            else:
                False
