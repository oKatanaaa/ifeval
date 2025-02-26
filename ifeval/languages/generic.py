"""Generic language-agnostic instruction implementations."""

import json
import random
import re
from typing import Dict, List, Optional, Union, Any

from ifeval.core.instructions import BaseInstruction

# Default values for instruction parameters
DEFAULT_NUM_PLACEHOLDERS = 4
DEFAULT_NUM_BULLETS = 5
DEFAULT_NUM_HIGHLIGHTS = 4
DEFAULT_NUM_PARAGRAPHS = 5

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