"""Command-line interface for the IFEval framework."""

import argparse
import json
import logging
import os
import sys
from typing import List, Dict, Any, Optional, Tuple

from ifeval.core.evaluation import Evaluator
from ifeval.core.registry import InstructionRegistry
from ifeval.languages.en.instructions import instruction_registry as en_registry
from ifeval.languages.ru.instructions import instruction_registry as ru_registry
from ifeval.utils.config import Config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate instruction following capabilities of language models."
    )
    
    parser.add_argument(
        "--input_data", 
        type=str, 
        required=True,
        help="Path to input data JSONL file containing prompts and instructions."
    )
    
    parser.add_argument(
        "--input_response_data", 
        type=str, 
        required=True,
        help="Path to JSONL file containing model responses."
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="Directory to save evaluation results."
    )
    
    parser.add_argument(
        "--language", 
        type=str, 
        default="en",
        choices=["en", "ru"],  # Will add more languages in the future
        help="Language to evaluate (currently only English is supported)."
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging."
    )
    
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to JSON config file."
    )
    
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Config:
    """Load configuration from args and config file.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        Configuration object.
    """
    # Start with default config
    config_dict = {
        "strict_mode": True,
        "verbose": args.verbose,
        "language": args.language,
    }
    
    # Override with config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            file_config = json.load(f)
            config_dict.update(file_config)
    
    # Override with command-line args (highest priority)
    if args.input_data:
        config_dict["input_data_path"] = args.input_data
    
    if args.input_response_data:
        config_dict["input_response_path"] = args.input_response_data
    
    if args.output_dir:
        config_dict["output_dir"] = args.output_dir
    
    return Config.from_dict(config_dict)


def get_registry_for_language(language: str) -> InstructionRegistry:
    """Get the instruction registry for a language.
    
    Args:
        language: Language code.
        
    Returns:
        Instruction registry for the language.
        
    Raises:
        ValueError: If the language is not supported.
    """
    if language == "en":
        return en_registry
    elif language == "ru":
        return ru_registry
    else:
        raise ValueError(f"Unsupported language: {language}")


def setup_logging(verbose: bool) -> None:
    """Set up logging.
    
    Args:
        verbose: Whether to enable verbose logging.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def run_evaluation(config: Config) -> Tuple[float, float]:
    """Run instruction following evaluation.
    
    Args:
        config: Configuration object.
        
    Returns:
        Tuple of (strict_accuracy, loose_accuracy)
    """
    # Create output directory if it doesn't exist
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Get registry for the specified language
    registry = get_registry_for_language(config.language)
    
    # Create evaluator
    evaluator = Evaluator(registry)
    
    # Run evaluation
    return evaluator.evaluate_dataset(
        config.input_data_path,
        config.input_response_path,
        config.output_dir
    )


def main() -> None:
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args)
    
    # Set up logging
    setup_logging(config.verbose)
    
    # Log configuration
    logging.info(f"Configuration: {config}")
    
    # Run evaluation
    strict_accuracy, loose_accuracy = run_evaluation(config)
    
    # Log results
    logging.info(f"Strict accuracy: {strict_accuracy:.4f}")
    logging.info(f"Loose accuracy: {loose_accuracy:.4f}")
    
    # Return success
    return 0


if __name__ == "__main__":
    sys.exit(main())