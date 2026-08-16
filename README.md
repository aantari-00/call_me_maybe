*This project has been created as part of the 42 curriculum by <aantari>.*

# call me maybe — Introduction to function calling in LLMs

## Description

This project turns a natural-language prompt (e.g. *"What is the sum of 2 and
3?"*) into a structured function call (e.g. `fn_add_numbers(a=2, b=3)`)
using a small local language model, **Qwen/Qwen3-0.6B**.

Small models are unreliable at producing valid JSON on their own. Instead of
prompting the model and hoping for well-formed output, this project uses
**constrained decoding**: at every generation step, the model is only
allowed to choose among the tokens that keep the output structurally and
semantically valid for the expected JSON schema. This guarantees 100% valid,
schema-compliant output, regardless of how well the model "wants" to behave.

The program reads a list of available functions and a list of prompts, and
writes one JSON object per prompt containing the chosen function name and
its arguments.

## Instructions

### Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for dependency management
- The `llm_sdk/` folder must sit next to `src/` at the project root (already
  the case in this repository)

### Install

```bash
make install
# equivalent to: uv sync
```

The first run will download the `Qwen/Qwen3-0.6B` model weights from the
Hugging Face Hub (a few hundred MB), so an internet connection is required
the first time.

### Run

```bash
make run
# equivalent to: uv run python -m src
```

By default this reads `data/input/functions_definition.json` and
`data/input/function_calling_tests.json`, and writes
`data/output/function_calling_results.json`.

Custom paths can be given explicitly:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

### Other Makefile targets

- `make debug` — run the program under Python's `pdb` debugger.
- `make lint` — run `flake8` and `mypy`.
- `make lint-strict` — run `flake8` and `mypy --strict`.
- `make clean` — remove `__pycache__` and `.mypy_cache`.

## Resources

- [Hugging Face — Text generation strategies](https://huggingface.co/docs/transformers/generation_strategies)
- [Hugging Face — Tokenizer summary (BPE, byte-level encoding)](https://huggingface.co/docs/transformers/tokenizer_summary)
- [JSON specification (RFC 8259)](https://www.rfc-editor.org/rfc/rfc8259)
- [Outlines — Structured generation with LLMs (background reading on constrained decoding)](https://github.com/outlines-dev/outlines)
- [pydantic documentation](https://docs.pydantic.dev/)

**How AI was used:** an AI assistant was used to help review the existing
constrained-decoding code (`generation.py`, `pipeline.py`), find and explain
two bugs (a wrong method name causing a crash on string parameters, and
missing schema enforcement letting integers receive a decimal point), and to
help write the project scaffolding (`__main__.py`'s argument parsing,
`Makefile`, `.gitignore`, this README). Every change was reviewed and
tested manually before being kept, and the original constrained-decoding
design (trie-based function name matching, digit-by-digit number
generation) was written and understood without AI assistance.

## Algorithm explanation

Generation happens token by token. At each step:

1. The model's logits (raw scores for every possible next token) are
   requested via `sdk.get_logits_from_input_ids(input_ids)`.
2. Depending on what is currently being generated, only a subset of tokens
   is *allowed*:
   - **Function name**: a trie is built from the token sequence of every
     candidate function name (plus a closing quote). At each step, only the
     children of the current trie node are allowed — this guarantees the
     model can only ever produce one of the known function names.
   - **Number**: only digit tokens are allowed, plus `-` as the very first
     token, plus `.` only for `"number"` parameters (never for
     `"integer"` parameters — this enforces the integer/float schema), plus
     a terminating token (`,`, `}` or space) once at least one digit has
     been produced.
   - **Boolean**: only the tokens for `true` and `false` are allowed.
   - **String**: characters are generated with a small bias toward the
     closing quote token, so strings terminate at a reasonable length.
3. Among the allowed tokens, the one with the highest logit is picked.
4. The chosen token id is appended to the sequence, and the process repeats.

Because the program builds the final answer as Python objects
(`FunctionCallResult` from `models.py`) and serializes them with the
standard `json` module at the very end, the output JSON is always
syntactically valid by construction — the model's choices only affect the
*content* of the fields, never the ability of the file to parse as JSON.

## Design decisions

- **Objects, not text.** The pipeline never manually assembles JSON text
  from generated tokens; it always builds Python values (`int`, `float`,
  `str`, `bool`, `dict`) and only serializes them once, using `json.dumps`.
  This removes an entire class of bugs (missing commas, unescaped quotes).
- **Trie-based function selection.** Restricting each step to the trie's
  current children is simpler and more robust than trying to score entire
  candidate strings.
- **One bad prompt does not fail the batch.** If constrained decoding fails
  for a single prompt (for example an unusually ambiguous one), the program
  logs a warning and continues with the remaining prompts, instead of
  crashing the whole run. This matches the "robust error handling" and
  "never crash unexpectedly" requirements from the subject.

## Performance analysis

- **Accuracy / validity:** because every generated token is constrained to
  the current JSON schema, output is always valid JSON and always contains
  a real function name and typed arguments — no post-hoc validation or
  retries are needed.
- **Speed:** each call to `get_logits_from_input_ids` runs a full forward
  pass of the 0.6B model. Generation is sequential (one token per call), so
  the overall speed is proportional to the number of generated tokens
  across all prompts, not to the number of prompts alone.
- **Reliability:** constrained decoding removes the small model's main
  weakness (producing malformed structured output) entirely, at the cost of
  some flexibility (e.g. it cannot invent a function that was not listed).

## Challenges faced

- `pipeline.py` called `vocab.token_id_for(...)`, a method that does not
  exist on `VocabIndex` (only `get_token_id` does). This crashed the
  program on every function with a string parameter. Fixed by using the
  correct method name.
- Number generation initially allowed a decimal point for `"integer"`
  parameters, which could silently break schema compliance. Fixed by only
  allowing `.` when the parameter type is `"number"`.

## Testing strategy

Because downloading and running the real model is slow, the core
constrained-decoding functions in `generation.py` were validated with small,
hand-written fake vocabularies and a fake SDK that returns fixed logits.
This made it possible to check, deterministically:

- an `"integer"` parameter never receives a `.` token, even when the fake
  model strongly "prefers" it;
- a `"number"` parameter can receive a `.`;
- boolean generation picks the higher-scoring of `true`/`false`;
- the function-name trie correctly follows the model's preferred branch;
- a full `pipeline.process_prompt()` call for a string parameter (the
  scenario that previously crashed) now returns the correct
  `FunctionCallResult`.

For an end-to-end check with the real model, run `make run` on the provided
`data/input/` files and inspect `data/output/function_calling_results.json`.

## Example usage

```bash
make install
make run
cat data/output/function_calling_results.json
```

Expected shape of the output (values depend on what the model actually
generates):

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {"s": "hello"}
  }
]
```
