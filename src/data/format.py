"""Convert xLAM data to Qwen2.5-Instruct chat template format.

Converts raw xLAM format ({query, tools, answers}) into the message format
expected by Qwen2.5's tokenizer: OpenAI-style tool definitions and
<tool_call> blocks in the assistant response.

Usage:
    python -m src.data.format \
        --input data/processed/xlam_4k.json \
        --output data/processed/xlam_4k_formatted.jsonl
"""

import argparse
import json
import re
from pathlib import Path

SYSTEM_PROMPT = "You are a helpful assistant with access to functions. Use them if required."

# xLAM type string -> JSON Schema type
TYPE_MAP = {
    "str": "string",
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
}

# Matches List[X] or list[X]
LIST_PATTERN = re.compile(r"[Ll]ist\[(.+)\]")


def xlam_type_to_json_schema(type_str: str) -> dict:
    """Convert an xLAM type string to a JSON Schema type dict.

    Examples:
        "str" -> {"type": "string"}
        "int" -> {"type": "integer"}
        "str, optional" -> {"type": "string"}
        "List[str]" -> {"type": "array", "items": {"type": "string"}}
    """
    # Strip ", optional" suffix - optionality is handled via `required`
    cleaned = type_str.split(",")[0].strip()

    # Check for List[X] pattern
    match = LIST_PATTERN.match(cleaned)
    if match:
        inner = match.group(1).strip()
        inner_type = TYPE_MAP.get(inner, "string")
        return {"type": "array", "items": {"type": inner_type}}

    return {"type": TYPE_MAP.get(cleaned, "string")}


def convert_tool(tool: dict) -> dict:
    """Convert one xLAM tool definition to OpenAI function-calling format."""
    properties = {}
    required = []

    for param_name, param_info in tool["parameters"].items():
        if not isinstance(param_info, dict):
            continue

        raw_type = param_info.get("type", "str")
        prop = xlam_type_to_json_schema(raw_type)

        if param_info.get("description"):
            prop["description"] = param_info["description"]

        is_optional = "optional" in raw_type.lower()
        if "default" in param_info:
            prop["default"] = param_info["default"]
        elif not is_optional:
            required.append(param_name)

        properties[param_name] = prop

    parameters = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required

    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": parameters,
        },
    }


def format_tool_call(answer: dict) -> str:
    """Format one answer as a <tool_call> block."""
    call = {"name": answer["name"], "arguments": answer["arguments"]}
    return f"<tool_call>\n{json.dumps(call)}\n</tool_call>"


def format_example(example: dict) -> dict:
    """Convert one xLAM example to chat template format."""
    tools = [convert_tool(t) for t in example["tools"]]

    assistant_content = "\n".join(format_tool_call(a) for a in example["answers"])

    # tools are passed separately to tokenizer.apply_chat_template(messages, tools=tools)
    # which injects them into the system prompt automatically - no need to embed here
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["query"]},
        {"role": "assistant", "content": assistant_content},
    ]

    return {
        "messages": messages,
        "tools": tools,
        "ground_truth": example["answers"],
    }


def format_dataset(examples: list[dict]) -> list[dict]:
    """Convert all examples and print stats."""
    formatted = [format_example(ex) for ex in examples]

    single = sum(1 for ex in examples if len(ex["answers"]) == 1)
    parallel = sum(1 for ex in examples if len(ex["answers"]) > 1)
    no_call = sum(1 for ex in examples if len(ex["answers"]) == 0)
    avg_tools = (sum(len(ex["tools"]) for ex in examples) / len(examples)) if examples else 0.0

    print(f"Formatted {len(formatted)} examples")
    print(f"  Single call: {single}")
    print(f"  Parallel calls: {parallel}")
    print(f"  No call: {no_call}")
    print(f"  Avg tools per example: {avg_tools:.1f}")

    return formatted


def main():
    parser = argparse.ArgumentParser(description="Format xLAM data for Qwen2.5 chat template")
    parser.add_argument("--input", default="data/processed/xlam_4k.json")
    parser.add_argument("--output", default="data/processed/xlam_4k_formatted.jsonl")
    args = parser.parse_args()

    with open(args.input) as f:
        examples = json.load(f)

    formatted = format_dataset(examples)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for ex in formatted:
            f.write(json.dumps(ex) + "\n")

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
