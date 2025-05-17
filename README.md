# IFEval: Instruction Following Evaluation Framework

IFEval is a framework for evaluating how well language models follow specific instructions when generating responses.

## Overview

This framework is a refactoring of the original instruction following evaluation code, which was cumbersome, hard to extend, and difficult to use. IFEval provides a clean, programmatic way (via both Python API and CLI) to evaluate model responses, with intuitive interfaces and elegant formatting of results.

IFEval also supports pass@k evaluation, including a smoothed estimator as described in the _Evaluating Large Language Models Trained on Code_ paper.

With IFEval, you can:

- Evaluate models on their ability to follow specific types of instructions
- Support multiple languages with a unified architecture
- Easily extend to new instruction types
- Use default benchmark datasets or your own custom data
- Compute pass@k and smoothed pass@k metrics (using the estimator from the _Evaluating Large Language Models Trained on Code_ paper)

Supported languages:
- English
- Russian

## Installation

```bash
# Clone the repository
git clone https://github.com/oKatanaaa/ifeval.git
cd ifeval

# Install the package
pip install .
```

## Usage

### Basic Usage (Python API)

```python
from ifeval import Evaluator, InputExample, instruction_registry, read_input_examples, read_responses

# For Russian instructions, use the Russian registry:
from ifeval import ru_instruction_registry

# Initialize evaluator
evaluator = Evaluator(instruction_registry)

# Load prompts and responses
input_examples = read_input_examples("path/to/prompts.jsonl")
responses = read_responses("path/to/responses.jsonl")

# Standard strict/loose evaluation
report, all_outputs = evaluator.evaluate(input_examples, responses)
print("Strict prompt accuracy:", report["eval_results_strict"]["prompt_accuracy"])
print("Loose prompt accuracy:", report["eval_results_loose"]["prompt_accuracy"])
```

### Pass@k Evaluation (Python API)

```python
from ifeval import read_responses_list

# Load multiple responses per prompt
responses_list = read_responses_list("path/to/responses.jsonl")

# Hard pass@k (any response follows all instructions)
report_hard, outputs_hard = evaluator.evaluate_pass_at_k_hard(input_examples, responses_list)
print("pass@k hard prompt accuracy:", report_hard["eval_results_strict"]["prompt_accuracy"])

# Smoothed pass@5 estimator
report_smooth, outputs_smooth = evaluator.evaluate_pass_at_k(input_examples, responses_list, k=5)
print("pass@5 smoothed prompt accuracy:", report_smooth["eval_results_loose"]["prompt_accuracy"])
```

### Using Default Datasets from HuggingFace
```python
from ifeval import Evaluator, instruction_registry, get_default_dataset

# For Russian datasets and instructions:
from ifeval import ru_instruction_registry
# input_examples_ru = get_default_dataset("ru")
# report_ru, outputs_ru = Evaluator(ru_instruction_registry).evaluate(input_examples_ru, responses)

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
# Basic evaluation (custom prompts/responses or default dataset)
python -m ifeval.cli \
    --input_data path/to/prompts.jsonl \
    --input_response_data path/to/responses.jsonl \
    --output_dir path/to/output_dir \
    --language en \
    --verbose

# pass@k evaluation (hard and smoothed) with multiple responses per prompt
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