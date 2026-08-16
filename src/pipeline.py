from llm_sdk import Small_LLM_Model
from src.models import FunctionSchema, PromptItem, FunctionCallResult
from src.vocab_index import VocabIndex
from src.generation import (
    create_function_prompt,
    add_literal_tokens,
    create_function_name_trie,
    generate_function_name_from_trie,
    generate_json_number,
    generate_json_string,
    generate_json_boolean,
)


def process_prompt(sdk: Small_LLM_Model, prompt_item: PromptItem,
                   functions: list[FunctionSchema],
                   vocab: VocabIndex) -> FunctionCallResult:
    """Process a single prompt into a validated function call result."""

    initial_text = create_function_prompt(prompt_item.prompt, functions)
    input_ids = sdk.encode(initial_text).flatten().tolist()

    function_names = [f.name for f in functions]
    trie = create_function_name_trie(sdk, function_names)
    fn_name, name_ids = generate_function_name_from_trie(sdk, input_ids,
                                                         trie, vocab)

    input_ids = input_ids + name_ids

    schema = next((f for f in functions if f.name == fn_name), None)
    if schema is None:
        raise ValueError(
            f"Generated function name '{fn_name}' not found in schema"
        )

    input_ids = add_literal_tokens(sdk, input_ids, ', "parameters": {')

    params: dict[str, object] = {}
    keys = list(schema.parameters.keys())

    for i, key in enumerate(keys):
        input_ids = add_literal_tokens(sdk, input_ids, f'"{key}":')
        param_type = schema.parameters[key].type

        value: object

        if param_type in ["number", "integer"]:
            value, value_ids = generate_json_number(
                sdk, input_ids, vocab, param_type
            )

        elif param_type == "string":
            input_ids = add_literal_tokens(sdk, input_ids, '"')
            value, value_ids = generate_json_string(sdk, input_ids, vocab)
            value_ids = value_ids + [vocab.get_token_id('"')]

        elif param_type == "boolean":
            value, value_ids = generate_json_boolean(sdk, input_ids, vocab)

        else:
            raise ValueError(f"Unsupported parameter type: {param_type}")

        params[key] = value
        input_ids = input_ids + value_ids

        if i < len(keys) - 1:
            input_ids = add_literal_tokens(sdk, input_ids, ", ")

    return FunctionCallResult(
        prompt=prompt_item.prompt,
        name=fn_name,
        parameters=params,
    )
