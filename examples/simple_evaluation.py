"""Simple example demonstrating how to use the IFEval framework."""

import json
import os

from ifeval.core.evaluation import Evaluator, InputExample
from ifeval.languages.en.instructions import instruction_registry
from ifeval.utils.config import Config


def create_example_data():
    """Create example data for demonstration.
    
    Returns:
        Tuple of (prompts_file, responses_file)
    """
    # Create example prompts
    prompts = [
        {
            "key": 1,
            "prompt": "Write a short story about a cat. Do not use any commas.",
            "instruction_id_list": ["punctuation:no_comma"],
            "kwargs": [{}]
        },
        {
            "key": 2,
            "prompt": "Write a blog post about AI with at least 200 words.",
            "instruction_id_list": ["length_constraints:number_words"],
            "kwargs": [{"relation": "at least", "num_words": 200}]
        },
        {
            "key": 3,
            "prompt": "Write a poem about the ocean. Use all lowercase letters.",
            "instruction_id_list": ["change_case:english_lowercase"],
            "kwargs": [{}]
        }
    ]
    
    # Create example responses (these would typically come from a model)
    responses = [
        {
            "prompt": "Write a short story about a cat. Do not use any commas.",
            "response": "Once upon a time there was a cat named Whiskers. Whiskers liked to play with yarn and chase mice. One day Whiskers found a ball of yarn and played with it all day long. The end."
        },
        {
            "prompt": "Write a blog post about AI with at least 200 words.",
            "response": "# The Future of AI\n\nArtificial Intelligence has come a long way in recent years. From simple rule-based systems to complex neural networks, the field has evolved rapidly. Today, AI is used in many applications, from voice assistants to autonomous vehicles.\n\nOne of the most exciting developments in AI is the emergence of large language models. These models can generate text, translate languages, and even write code. They are trained on massive datasets and can perform a wide range of tasks.\n\nHowever, there are also concerns about the impact of AI on society. Some worry about job displacement, while others raise ethical questions about bias and privacy. These are important issues that need to be addressed as AI continues to advance.\n\nIn conclusion, AI has the potential to revolutionize many aspects of our lives, but it must be developed responsibly. As we move forward, it will be crucial to balance innovation with ethical considerations."
        },
        {
            "prompt": "Write a poem about the ocean. Use all lowercase letters.",
            "response": "the deep blue sea\nwaves crashing on the shore\nsalt in the air\nsun on my skin\n\nthe ocean's vastness\ncalls to me\nlike an old friend\nwelcoming me home"
        }
    ]
    
    # Create directory for example data
    os.makedirs("example_data", exist_ok=True)
    
    # Write prompts to file
    prompts_file = "example_data/prompts.jsonl"
    with open(prompts_file, "w") as f:
        for prompt in prompts:
            f.write(json.dumps(prompt) + "\n")
    
    # Write responses to file
    responses_file = "example_data/responses.jsonl"
    with open(responses_file, "w") as f:
        for response in responses:
            f.write(json.dumps(response) + "\n")
    
    return prompts_file, responses_file


def main():
    """Run example evaluation."""
    # Create example data
    prompts_file, responses_file = create_example_data()
    
    # Load data from files
    from ifeval.utils.io import read_input_examples, read_responses, write_outputs
    input_examples = read_input_examples(prompts_file)
    responses = read_responses(responses_file)
    
    # Create output directory
    output_dir = "example_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create evaluator
    evaluator = Evaluator(instruction_registry)
    
    # Run evaluation
    report, all_outputs = evaluator.evaluate(input_examples, responses)
    
    # Print report
    evaluator.print_report(report)
    
    # Write outputs to files
    for output_key, outputs in all_outputs.items():
        output_path = os.path.join(output_dir, f"{output_key}.jsonl")
        write_outputs(output_path, outputs)
        print(f"Wrote {len(outputs)} outputs to {output_path}")
    
    # Extract accuracies
    strict_accuracy = report["eval_results_strict"]["prompt_accuracy"]
    loose_accuracy = report["eval_results_loose"]["prompt_accuracy"]
    
    # Print summary
    print(f"\nEvaluation Results:")
    print(f"Strict accuracy: {strict_accuracy:.4f}")
    print(f"Loose accuracy: {loose_accuracy:.4f}")
    print(f"\nDetailed results saved to {output_dir}")



if __name__ == "__main__":
    main()