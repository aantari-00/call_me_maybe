from llm_sdk import Small_LLM_Model

from src.vocab_index import VocabIndex


def add_literal_tokens(
    sdk: Small_LLM_Model,
    current_ids: list[int],
    text: str
) -> list[int]:
    """Encode text and add its tokens to the current sequence."""
    encoded = sdk.encode(text)
    token_ids: list[int] = encoded.flatten().tolist()
    return current_ids + token_ids


def choose_best_allowed_token(
    sdk: Small_LLM_Model,
    input_ids: list[int],
    allowed_ids: set[int]
) -> int:
    """Choose the allowed token with the highest model score."""
    if not allowed_ids:
        raise ValueError("allowed_ids must not be empty")

    logits = sdk.get_logits_from_input_ids(input_ids)

    best_token_id = -1
    best_score = float("-inf")

    for token_id in allowed_ids:
        score = logits[token_id]

        if score > best_score:
            best_score = score
            best_token_id = token_id

    return best_token_id


def generate_json_number(
    sdk: Small_LLM_Model,
    input_ids: list[int],
    vocab: VocabIndex,
    param_type: str,
    max_tokens: int = 20
) -> tuple[float, list[int]]:
    """Generate a JSON number using constrained decoding."""

    digit_ids = vocab.get_numeric_token_ids()
    dot_id = vocab.get_token_id(".")
    minus_id = vocab.get_token_id("-")
    end_ids = vocab.get_number_end_token_ids()

    generated_ids: list[int] = []
    has_decimal_point = False

    for _ in range(max_tokens):
        allowed_ids = set(digit_ids)

        if param_type != "integer" and not has_decimal_point and dot_id != -1:
            allowed_ids.add(dot_id)

        if len(generated_ids) == 0 and minus_id != -1:
            allowed_ids.add(minus_id)

        if len(generated_ids) > 0:
            allowed_ids.update(end_ids)

        next_id = choose_best_allowed_token(
            sdk,
            input_ids + generated_ids,
            allowed_ids
        )

        if next_id in end_ids:
            break

        if next_id == dot_id:
            has_decimal_point = True

        generated_ids.append(next_id)

    number_text = ""

    for token_id in generated_ids:
        number_text += vocab.id_to_token[token_id]

    if param_type == "integer":
        return int(number_text), generated_ids
    return float(number_text), generated_ids


def generate_json_string(
    sdk: Small_LLM_Model,
    input_ids: list[int],
    vocab: VocabIndex,
    max_tokens: int = 20,
    stop_bias: float = 4.0
) -> tuple[str, list[int]]:
    """Generate a string value token by token."""

    quote_id = vocab.get_token_id('"')

    if quote_id == -1:
        raise ValueError("Quote character not found in vocab")

    generated_ids: list[int] = []
    generated_text = ""
    previous_char_is_backslash = False

    for _ in range(max_tokens):
        current_ids = input_ids + generated_ids
        logits = sdk.get_logits_from_input_ids(current_ids)

        if len(generated_ids) > 0:
            logits[quote_id] += stop_bias

        best_id = 0
        best_score = logits[0]

        for token_id in range(1, len(logits)):
            if logits[token_id] > best_score:
                best_score = logits[token_id]
                best_id = token_id

        token = vocab.id_to_token[best_id]
        token = token.replace("Ġ", " ")

        reached_end_of_string = False

        for char in token:
            if previous_char_is_backslash:
                generated_text += char
                previous_char_is_backslash = False
                continue

            if char == "\\":
                previous_char_is_backslash = True
                continue

            if char == '"':
                reached_end_of_string = True
                break

            generated_text += char

        if reached_end_of_string:
            break

        generated_ids.append(best_id)

    generated_text = generated_text.strip()

    return generated_text, generated_ids


def generate_json_boolean(
    sdk: Small_LLM_Model,
    input_ids: list[int],
    vocab: VocabIndex
) -> tuple[bool, list[int]]:
    """Generate either true or false using constrained decoding."""

    true_id = vocab.get_token_id("true")
    false_id = vocab.get_token_id("false")

    if true_id == -1 or false_id == -1:
        raise ValueError("true/false tokens not found in vocab")

    allowed_ids = {true_id, false_id}

    chosen_id = choose_best_allowed_token(
        sdk,
        input_ids,
        allowed_ids
    )

    if chosen_id == true_id:
        return True, [chosen_id]

    return False, [chosen_id]


def create_function_name_trie(
    sdk: Small_LLM_Model,
    function_names: list[str]
) -> dict:
    """Create a trie containing the token sequence of each function name."""

    trie: dict = {}

    for name in function_names:
        text = name + '"'
        encoded = sdk.encode(text)
        token_ids = encoded.flatten().tolist()

        current_node = trie

        for token_id in token_ids:
            if token_id not in current_node:
                current_node[token_id] = {}

            current_node = current_node[token_id]

        current_node["__end__"] = True

    return trie


def generate_function_name_from_trie(
    sdk: Small_LLM_Model,
    input_ids: list[int],
    trie: dict,
    vocab: VocabIndex
) -> tuple[str, list[int]]:
    """Generate a function name while following the trie."""

    current_node = trie
    generated_ids: list[int] = []

    while "__end__" not in current_node:

        allowed_ids = set()

        for token_id in current_node:
            if token_id != "__end__":
                allowed_ids.add(token_id)

        if not allowed_ids:
            raise ValueError("Trie node has no valid continuations")

        next_id = choose_best_allowed_token(
            sdk,
            input_ids + generated_ids,
            allowed_ids
        )

        generated_ids.append(next_id)
        current_node = current_node[next_id]

    name_text = ""

    for token_id in generated_ids:
        name_text += vocab.id_to_token[token_id]

    name_text = name_text.lstrip("Ġ")
    name_text = name_text.strip('"')

    return name_text, generated_ids


def create_function_prompt(
    user_prompt: str,
    functions: list
) -> str:
    """Build the prompt containing the available functions."""

    function_lines = []

    for function in functions:
        parameter_names = ", ".join(function.parameters.keys())

        parameter_types = []

        for name, parameter in function.parameters.items():
            parameter_types.append(f"{name}: {parameter.type}")

        types_text = ", ".join(parameter_types)

        line = (
            f"- {function.name}({parameter_names}): "
            f"{function.description} "
            f"(params: {types_text})"
        )

        function_lines.append(line)

    functions_text = "\n".join(function_lines)

    prompt = (
        "<|im_start|>system\n"
        "You are an expert function caller. "
        "Select the correct tool and fill parameters exactly.\n\n"
        "Available tools:\n"
        f"{functions_text}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_prompt}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        'Let me select the correct function:\n'
        '{"name": "'
    )

    return prompt
