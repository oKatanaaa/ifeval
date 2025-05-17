# Developer Guide

# Instruction Types

The following instruction types are supported out of the box:

- **Language**:
  - `language:response_language`: Check response language

- **Length Constraints**:
  - `length_constraints:number_sentences`: Check number of sentences
  - `length_constraints:number_paragraphs`: Check number of paragraphs
  - `length_constraints:number_words`: Check number of words
  - `length_constraints:nth_paragraph_first_word`: Check first word of specific paragraph

- **Content Detection**:
  - `detectable_content:number_placeholders`: Check number of placeholders
  - `detectable_content:postscript`: Check for postscript

- **Format**:
  - `detectable_format:number_bullet_lists`: Check number of bullet points
  - `detectable_format:constrained_response`: Check constrained response
  - `detectable_format:number_highlighted_sections`: Check highlighted sections
  - `detectable_format:multiple_sections`: Check multiple sections
  - `detectable_format:json_format`: Check JSON format
  - `detectable_format:title`: Check title format

- **Combinations**:
  - `combination:two_responses`: Check for two separate responses
  - `combination:repeat_prompt`: Check response repeats prompt

- **Start/End Patterns**:
  - `startend:end_checker`: Check response ends with specific phrase
  - `startend:quotation`: Check response wrapped in quotes

- **Case**:
  - `change_case:capital_word_frequency`: Check frequency of capitalized words
  - `change_case:english_capital`: Check all capital letters
  - `change_case:english_lowercase`: Check all lowercase letters

- **Punctuation**:
  - `punctuation:no_comma`: Check absence of commas

- **Keywords**:
  - `keywords:existence`: Check for existence of specific keywords
  - `keywords:frequency`: Check frequency of specific keywords
  - `keywords:forbidden_words`: Check absence of forbidden words
  - `keywords:letter_frequency`: Check frequency of specific letters

# Extending the Framework

## Adding New Instruction Types

### Generic Instructions

For instruction types that are language-agnostic, add them to the generic module:

```python
from ifeval.core.instructions import BaseInstruction

class MyGenericInstruction(BaseInstruction):
    def __init__(self, parameter1, parameter2):
        # Store parameters
        self._parameter1 = parameter1
        self._parameter2 = parameter2
        
    def check_following(self, value):
        # Check if response follows instruction
        # Use language-agnostic logic (e.g., regex patterns)
        # ...
        return True/False
```

Then register it in each language-specific instruction module:

```python
# In ifeval/languages/en/instructions.py and ifeval/languages/ru/instructions.py
from ifeval.languages.generic import MyGenericInstruction

instruction_registry.register("my_category:my_instruction")(MyGenericInstruction)
```

> P.S. You'll see `get_instruction_args` and `get_instruction_args_keys` methods in existing implementations.
This is legacy API, don't bother to implement it.

## Language-Specific Instructions

For instructions that need language-specific processing:

```python
from ifeval.core.instructions import BaseInstruction

class MyLanguageSpecificInstruction(BaseInstruction):
    def __init__(self, parameter1, parameter2):
        # Store parameters
        self._parameter1 = parameter1
        self._parameter2 = parameter2
        
    def check_following(self, value):
        # Use language processor to analyze the text if needed
        # processor.count_words(value), processor.lemmatize(text), etc.
        # ...
        return True/False
```

Register in the specific language module:

```python
from ifeval.core.registry import InstructionRegistry

# Specific language registry (already defined in language module)
instruction_registry.register("my_category:my_instruction")(MyLanguageSpecificInstruction)
```

## Adding New Languages

1. Create a language processor inheriting from `BaseLanguageProcessor`:

```python
from ifeval.languages.language_processor import BaseLanguageProcessor

class MyLanguageProcessor(BaseLanguageProcessor):
    def count_sentences(self, text):
        # Count sentences using language-specific rules
        
    def count_words(self, text):
        # Count words using language-specific tokenization
        
    def split_into_sentences(self, text):
        # Split text into sentences
        
    def lemmatize(self, text):
        # Lemmatize words according to language rules
        
    def word_tokenize(self, text):
        # Tokenize text into words
```

2. Register the processor:

```python
from ifeval.languages.language_registry import LanguageRegistry

language_registry = LanguageRegistry()
language_registry.register("my_lang")(MyLanguageProcessor)
```

3. Create a language-specific constants module:

```python
# ifeval/languages/my_lang/constants.py

# Define language-specific constants
CONSTRAINED_RESPONSE_OPTIONS = ("My answer is yes.", "My answer is no.")  # Translated to your language
```

4. Create a language-specific instructions module:

```python
# ifeval/languages/my_lang/instructions.py

from ifeval.core.registry import InstructionRegistry
from ifeval.languages.my_lang.processor import MyLanguageProcessor
from ifeval.languages.my_lang.constants import COMPARISON_RELATION, CONSTRAINED_RESPONSE_OPTIONS
from ifeval.languages.generic import (
    PlaceholderChecker,
    BulletListChecker,
    # Import all generic instruction classes
)

# Create registry and processor instances
instruction_registry = InstructionRegistry()
processor = MyLanguageProcessor()

# Define instruction type prefixes
_KEYWORD = "keywords:"
_LANGUAGE = "language:"
# Define all prefix constants

# Register generic instructions
instruction_registry.register(_CONTENT + "number_placeholders")(PlaceholderChecker)
instruction_registry.register(_FORMAT + "multiple_sections")(SectionChecker)
# Register all generic instructions

# Implement language-specific instructions
# ...
```

5. Register the language in the main package init file.