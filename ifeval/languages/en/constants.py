"""Constants for English language processing."""

import string
import random

# The relational operation for comparison.
COMPARISON_RELATION = ("less than", "at least")

# The maximum number of sentences.
MAX_NUM_SENTENCES = 20

# The number of placeholders.
NUM_PLACEHOLDERS = 4

# The number of bullet lists.
NUM_BULLETS = 5

# The options of constrained response.
CONSTRAINED_RESPONSE_OPTIONS = (
    "My answer is yes.", "My answer is no.", "My answer is maybe."
)

# The options of starter keywords.
STARTER_OPTIONS = (
    "I would say", "My answer is", "I believe",
    "In my opinion", "I think", "I reckon", "I feel",
    "From my perspective", "As I see it", "According to me",
    "As far as I'm concerned", "To my understanding",
    "In my view", "My take on it is", "As per my perception"
)

# The options of ending keywords.
ENDING_OPTIONS = (
    "Any other questions?",
    "Is there anything else I can help with?"
)

# The number of highlighted sections.
NUM_HIGHLIGHTED_SECTIONS = 4

# The section spliter.
SECTION_SPLITER = ("Section", "SECTION")

# The number of sections.
NUM_SECTIONS = 5

# The number of paragraphs.
NUM_PARAGRAPHS = 5

# The postscript marker.
POSTSCRIPT_MARKER = ("P.S.", "P.P.S")

# The number of keywords.
NUM_KEYWORDS = 2

# The occurrences of a single keyword.
KEYWORD_FREQUENCY = 3

# The occurrences of a single letter.
LETTER_FREQUENCY = 10

# The occurrences of words with all capital letters.
ALL_CAPITAL_WORD_FREQUENCY = 20

# The number of words in the response.
NUM_WORDS_LOWER_LIMIT = 100
NUM_WORDS_UPPER_LIMIT = 500

# Common English words for testing
WORD_LIST = ["western", "sentence", "signal", "dump", "spot", "opposite", "bottom", "potato", 
             "administration", "working", "welcome", "morning", "good", "agency", "primary", 
             "wish", "responsibility", "press", "problem", "president", "steal", "brush", 
             "read", "type", "beat", "trainer", "growth", "lock", "bone", "case", "equal", 
             "comfortable", "region", "replacement", "performance", "mate", "walk", "medicine", 
             "film", "thing", "rock", "tap", "total", "competition", "ease", "south", 
             "establishment", "gather", "parking", "world", "plenty", "breath", "claim", 
             "alcohol", "trade", "dear", "highlight", "street", "matter", "decision", "mess", 
             "agreement", "studio", "coach", "assist", "brain", "wing", "style", "private", 
             "top", "brown", "leg", "buy", "procedure", "method", "speed", "high", "company"]


def generate_keywords(num_keywords=NUM_KEYWORDS):
    """Generate random keywords for testing.
    
    Args:
        num_keywords: Number of keywords to generate.
        
    Returns:
        A list of keywords.
    """
    return random.sample(WORD_LIST, k=min(num_keywords, len(WORD_LIST)))