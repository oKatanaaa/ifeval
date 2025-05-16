"""Example demonstrating pass@k evaluation with IFEval."""

import json
import os

from ifeval.core.evaluation import Evaluator
from ifeval.languages.en.instructions import instruction_registry
from ifeval.utils.io import read_input_examples, read_responses_list


def create_pass_at_k_example_data():
    """Create example prompts and multiple responses for pass@k demonstration."""
    prompts = [
        {
            "key": 1,
            "prompt": "Write a sentence with no commas.",
            "instruction_id_list": ["punctuation:no_comma"],
            "kwargs": [{}]
        }
    ]
    responses = [
        {
            "prompt": "Write a sentence with no commas.",
            "responses": [
                "This sentence contains, a comma.",
                "This sentence contains no commas"
            ]
        }
    ]
    os.makedirs("pass_at_k_data", exist_ok=True)
    prompts_file = "pass_at_k_data/prompts.jsonl"
    responses_file = "pass_at_k_data/responses.jsonl"
    with open(prompts_file, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(responses_file, "w", encoding="utf-8") as f:
        for r in responses:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return prompts_file, responses_file


def main():
    prompts_file, responses_file = create_pass_at_k_example_data()
    input_examples = read_input_examples(prompts_file)
    responses = read_responses_list(responses_file)
    evaluator = Evaluator(instruction_registry)
    # Hard estimator (k = number of provided responses)
    n = len(responses[next(iter(responses))])
    report_hard, outputs_hard = evaluator.evaluate_pass_at_k_hard(input_examples, responses)
    hard_score = report_hard["eval_results_strict"]["prompt_accuracy"]
    print(f"Hard pass@{n}: {hard_score:.4f}")

    # Smooth estimator for different k values
    for k in [1, 2]:
        report_smooth, outputs_smooth = evaluator.evaluate_pass_at_k(input_examples, responses, k=k)
        smooth_score = report_smooth["eval_results_loose"]["prompt_accuracy"]
        print(f"Smooth pass@{k}: {smooth_score:.4f}")


if __name__ == "__main__":
    main()