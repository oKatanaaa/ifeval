# IFEval: Instruction Following Evaluation Framework

IFEval is a framework for evaluating how well language models follow specific instructions when generating responses.

## Overview

This framework is a refactoring of the original instruction following evaluation code, which was cumbersome, hard to extend, and difficult to use. IFEval provides a clean, programmatic way (via both Python API and CLI) to evaluate model responses, with intuitive interfaces and elegant formatting of results.

With IFEval, you can:

- Evaluate models on their ability to follow specific types of instructions
- Support multiple languages with a unified architecture
- Easily extend to new instruction types
- Run both strict and loose evaluations
- Use default benchmark datasets or your own custom data

## Installation

```bash
# Clone the repository
git clone https://github.com/oKatanaaa/ifeval.git
cd ifeval

# Install the package
pip install .
```

## Requirements

- Python 3.8+
- Dependencies:
  - langdetect
  - nltk
  - immutabledict
  - absl-py
  - datasets (for loading default evaluation datasets)

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
from ifeval.datasets import get_default_dataset

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

## Extending the Framework

### Adding New Instruction Types

Create a new instruction class inheriting from `BaseInstruction`:

```python
from ifeval.core.instructions import BaseInstruction

class MyNewInstruction(BaseInstruction):
    def build_description(self, **kwargs):
        # Store parameters and build description
        # ...
        return description
        
    def get_instruction_args(self):
        # Return stored parameters
        # ...
        
    def get_instruction_args_keys(self):
        # Return list of parameter keys
        # ...
        
    def check_following(self, value):
        # Check if response follows instruction
        # ...
        return True/False
```

Register the instruction:

```python
from ifeval.core.registry import InstructionRegistry

registry = InstructionRegistry()
registry.register("my_category:my_instruction")(MyNewInstruction)
```

### Adding New Languages

Create a language processor inheriting from `BaseLanguageProcessor`:

```python
from ifeval.languages.language_processor import BaseLanguageProcessor

class MyLanguageProcessor(BaseLanguageProcessor):
    def detect_language(self, text):
        # ...
        
    def count_sentences(self, text):
        # ...
        
    def count_words(self, text):
        # ...
        
    def split_into_sentences(self, text):
        # ...
        
    def lemmatize(self, text):
        # ...
```

Register the processor:

```python
from ifeval.languages.language_registry import LanguageRegistry

language_registry = LanguageRegistry()
language_registry.register("my_lang")(MyLanguageProcessor)
```

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