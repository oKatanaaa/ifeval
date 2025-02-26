"""Constants for Russian language processing."""

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
    "Мой ответ — да",
    "Мой ответ — нет",
    "Мой ответ — возможно",
)

# The options of starter keywords.
STARTER_OPTIONS = (
    "Я бы сказал",
    "Мой ответ",
    "Я считаю",
    "По моему мнению",
    "Я думаю",
    "С моей точки зрения",
    "Насколько я понимаю",
)

# The options of ending keywords.
ENDING_OPTIONS = (
    "Есть еще какие-нибудь вопросы?", 
    "Могу ли я ещё чем-нибудь помочь?"
)

# The number of highlighted sections.
NUM_HIGHLIGHTED_SECTIONS = 4

# The section spliter.
SECTION_SPLITER = ("Раздел", "РАЗДЕЛ")

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

# Common Russian words for testing
WORD_LIST = ["время", "человек", "дело", "жизнь", "день", "рука", "раз", "работа", 
             "слово", "место", "лицо", "друг", "глаз", "вопрос", "дом", "сторона", 
             "страна", "мир", "случай", "голова", "ребенок", "сила", "конец", "вид", 
             "система", "часть", "город", "отношение", "женщина", "деньги", "земля", 
             "машина", "вода", "отец", "проблема", "час", "право", "нога", "решение", 
             "дверь", "образ", "история", "власть", "закон", "война", "бог", "голос", 
             "тысяча", "книга", "возможность", "результат", "ночь", "стол", "имя", 
             "область", "статья", "число", "компания", "народ", "жена", "группа", 
             "развитие", "процесс", "суд", "условие", "средство", "начало", "свет", 
             "пора", "путь", "душа", "уровень", "форма", "связь", "минута", "улица", 
             "вечер", "качество", "мысль", "дорога", "мать", "действие", "месяц", 
             "государство", "язык", "любовь", "взгляд", "мама", "век", "школа", 
             "цель", "общество", "деятельность", "организация", "президент", "комната"]


def generate_keywords(num_keywords=NUM_KEYWORDS):
    """Generate random keywords for testing.
    
    Args:
        num_keywords: Number of keywords to generate.
        
    Returns:
        A list of keywords.
    """
    return random.sample(WORD_LIST, k=min(num_keywords, len(WORD_LIST)))