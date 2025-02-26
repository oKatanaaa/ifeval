"""Language-agnostic text processing utilities."""

import re
from typing import List


def split_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs.
    
    Args:
        text: The text to split.
        
    Returns:
        A list of paragraphs.
    """
    return [p for p in re.split(r'\n\s*\n', text) if p.strip()]


def count_words_simple(text: str) -> int:
    """Count words in text (simple version).
    
    This is a simple word counter that just splits on whitespace.
    Use language-specific counters for more accurate results.
    
    Args:
        text: The text to count words in.
        
    Returns:
        The number of words.
    """
    return len(re.findall(r'\b\w+\b', text))


def remove_markdown(text: str) -> str:
    """Remove markdown formatting from text.
    
    Args:
        text: The text to process.
        
    Returns:
        Text with markdown formatting removed.
    """
    # Remove bold/italic markers
    text = re.sub(r'\*\*?(.*?)\*\*?', r'\1', text)
    
    # Remove headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Remove code blocks
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Remove blockquotes
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    
    # Remove horizontal rules
    text = re.sub(r'^-{3,}|^\*{3,}|^_{3,}', '', text, flags=re.MULTILINE)
    
    # Remove links
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
    # Remove images
    text = re.sub(r'!\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
    return text


def is_json(text: str) -> bool:
    """Check if text is valid JSON.
    
    Args:
        text: The text to check.
        
    Returns:
        True if the text is valid JSON, False otherwise.
    """
    import json
    text = text.strip()
    
    # Remove markdown code block syntax if present
    if text.startswith('```json') or text.startswith('```JSON'):
        text = text[7:]
    elif text.startswith('```'):
        text = text[3:]
    
    if text.endswith('```'):
        text = text[:-3]
    
    text = text.strip()
    
    try:
        json.loads(text)
        return True
    except:
        return False