import json
from pathlib import Path
import sys

from src.models import FunctionSchema, PromptItem, FunctionCallResult


def load_functions_definition(path: str) -> list[FunctionSchema]:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"the file : {path} not exists")

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as e:
        raise ValueError("Invalid JSON format in the file") from e

    if not isinstance(data, list):
        raise ValueError(f"Invalid data format in {path}. Expected a list.")

    functions: list[FunctionSchema] = []
    for item in data:
        try:
            functions.append(FunctionSchema(**item))
        except Exception as e:
            raise ValueError(
                f"Invalid function definition in {path}: {e}"
            ) from e

    return functions


def load_prompts(path: str) -> list[PromptItem]:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Test prompts file not found: {path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e

    if not isinstance(raw_data, list):
        raise ValueError(f"{path} must contain a JSON array")

    return [PromptItem(**item) for item in raw_data]


def save_results(path: str, results: list[FunctionCallResult]) -> None:
    file_path = Path(path)

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)

        data = [result.model_dump() for result in results]
        file_path.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8"
        )

    except OSError as e:
        print(f"Error: could not save results to {path}: {e}")
        sys.exit(1)
