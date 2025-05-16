# IFEval: Instruction Following Evaluation Framework

IFEval is a framework for evaluating how well language models follow specific instructions when generating responses.

## Overview

This framework is a refactoring of the original instruction following evaluation code, which was cumbersome, hard to extend, and difficult to use. IFEval provides a clean, programmatic way (via both Python API and CLI) to evaluate model responses, with intuitive interfaces and elegant formatting of results.

With IFEval, you can:

- Evaluate models on their ability to follow specific types of instructions
- Support multiple languages with a unified architecture
- Easily extend to new instruction types
- Use default benchmark datasets or your own custom data

## Installation

```bash
# Clone the repository
git clone https://github.com/oKatanaaa/ifeval.git
cd ifeval

# Install the package
pip install .
```

## Usage

### Basic Usage

```python
from ifeval.core.evaluation import Evaluator, InputExample
from ifeval.languages.en.instructions import instruction_registry
from ifeval.utils.io import read_input_examples, read_responses

# Create evaluator
evaluator = Evaluator(instruction_registry)

# Load input examples and responses
input_examples = read_input_examples("path/to/prompts.jsonl")
responses = read_responses("path/to/responses.jsonl")

# Run evaluation
report, all_outputs = evaluator.evaluate(input_examples, responses)

# Print report
evaluator.print_report(report)

# Access accuracies
strict_accuracy = report["eval_results_strict"]["prompt_accuracy"]
loose_accuracy = report["eval_results_loose"]["prompt_accuracy"]

print(f"Strict accuracy: {strict_accuracy}")
print(f"Loose accuracy: {loose_accuracy}")
```

### Using Default Datasets from HuggingFace
```python
from ifeval.core.evaluation import Evaluator
from ifeval.languages.en.instructions import instruction_registry
from ifeval.utils.huggingface import get_default_dataset

# Create evaluator
evaluator = Evaluator(instruction_registry)

# Load default dataset (English by default)
input_examples = get_default_dataset("en")

# Get responses from your model (example)
responses = {ex.prompt: your_model.generate(ex.prompt) for ex in input_examples}

# Run evaluation
report, all_outputs = evaluator.evaluate(input_examples, responses)
```

### Command-line Interface

```bash
# Using custom data
python -m ifeval.cli \
    --input_data path/to/prompts.jsonl \
    --input_response_data path/to/responses.jsonl \
    --output_dir path/to/output_dir \
    --language en \
    --verbose

# Using default datasets
python -m ifeval.cli \
    --input_response_data path/to/responses.jsonl \
    --output_dir path/to/output_dir \
    --language ru \  # Use Russian dataset
    --verbose

# pass@k evaluation with multiple responses per prompt
python -m ifeval.cli \
    --input_data path/to/prompts.jsonl \
    --input_response_data path/to/responses_with_lists.jsonl \
    --output_dir path/to/output_dir \
    --language en \
    --pass_k_hard \
    --pass_k 5 \
    --verbose
```

### Data Format

#### Input Data

The input data should be a JSONL file with each line containing a JSON object with the following fields:

```json
{
  "key": 1000,
  "prompt": "Write a 300+ word summary...",
  "instruction_id_list": ["punctuation:no_comma", "length_constraints:number_words"],
  "kwargs": [{}, {"relation": "at least", "num_words": 300}]
}
```

#### Response Data

The response data should be a JSONL file with each line containing a JSON object with the following fields:

```json
{
  "prompt": "Write a 300+ word summary...",
  "response": "This is the model's response..."
}
```

You can also supply multiple responses per prompt to compute pass@k metrics:

```json
{
  "prompt": "Write a 300+ word summary...",
  "responses": ["This is response 1...", "This is response 2...", ...]
}
```

## Instruction Types

IFEval currently supports the following instruction types for English:

- **Keywords**:
  - `keywords:existence`: Check for existence of specific keywords
  - `keywords:frequency`: Check frequency of specific keywords
  - `keywords:forbidden_words`: Check absence of forbidden words
  - `keywords:letter_frequency`: Check frequency of specific letters

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

## pass@k Evaluation

pass@k measures the probability that at least one out of k sampled responses follows all instructions.

### Python API

```python
from ifeval.core.evaluation import Evaluator
from ifeval.languages.en.instructions import instruction_registry
from ifeval.utils.io import read_input_examples, read_responses_list

evaluator = Evaluator(instruction_registry)
input_examples = read_input_examples("prompts.jsonl")
responses = read_responses_list("responses.jsonl")

hard_score = evaluator.evaluate_pass_at_k_hard(input_examples, responses)
smooth_score = evaluator.evaluate_pass_at_k(input_examples, responses, k=5)
print(f"Hard pass@5: {hard_score:.4f}")
print(f"Smooth pass@5: {smooth_score:.4f}")
```

See `examples/pass_at_k_example.py` for a runnable demonstration.

## Extending the Framework

### Adding New Instruction Types

#### Generic Instructions

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

> P.S. You'll see  `get_instruction_args` and `get_instruction_args_keys` methods in existing implementations.
This is legacy API, don't bother to implement it.

#### Language-Specific Instructions

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

### Adding New Languages

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
CONSTRAINED_RESPONSE_OPTIONS = ("My answer is yes.", "My answer is no.")# Translated to your language
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

## Evaluation Methods

### Strict Evaluation

The strict evaluation checks if the response follows all instructions exactly as specified.

### Loose Evaluation

The loose evaluation applies various transformations to the response to see if any variant follows the instructions, such as:
- Removing the first/last line
- Replacing certain characters
- Combinations of the above

## Acknowledgements

This project is a refactoring and extension of the following works:

- [google-research/instruction_following_eval](https://github.com/google-research/google-research/tree/master/instruction_following_eval) - The original instruction following evaluation codebase developed by Google Research
- [NLP-Core-Team/ruIFEval](https://github.com/NLP-Core-Team/ruIFEval) - Russian version of IFEval that provided the Russian evaluation dataset and instructions

## License

This project is licensed under the Apache 2.0 License.