import argparse
import sys
from llm_sdk import Small_LLM_Model
from src.json_in_out import (load_functions_definition,
                             load_prompts,
                             save_results)
from src.models import FunctionCallResult
from src.pipeline import process_prompt
from src.vocab_index import VocabIndex


def parse_arguments() -> argparse.Namespace:
    """Parse the command line arguments.
    Returns:
        The parsed arguments, with sensible defaults matching
        data/input/ and data/output/.
    """
    parser = argparse.ArgumentParser(
        description="Translate natural language prompts into function calls."
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the JSON file describing the available functions.",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the JSON file containing the prompts to process.",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path where the JSON results will be written.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full function-calling pipeline."""
    args = parse_arguments()

    try:
        functions = load_functions_definition(args.functions_definition)
        prompts = load_prompts(args.input)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: could not load input files: {error}")
        sys.exit(1)

    print(f"Loaded {len(functions)} functions.")
    print(f"Loaded {len(prompts)} prompts.")

    try:
        sdk = Small_LLM_Model()
        vocab = VocabIndex(sdk.get_path_to_vocab_file())
    except Exception as error:
        print(f"Error: could not load the model: {error}")
        sys.exit(1)

    results: list[FunctionCallResult] = []

    for prompt_item in prompts:
        try:
            result = process_prompt(sdk, prompt_item, functions, vocab)
            results.append(result)
            print(f"OK: '{prompt_item.prompt}' -> "
                  f"{result.name}({result.parameters})")
        except Exception as error:
            print(f"Warning: could not process prompt '{prompt_item.prompt}': "
                  f"{error}")

    print(f"Saved {len(results)} results to {args.output}")
    save_results(args.output, results)


if __name__ == "__main__":
    main()
