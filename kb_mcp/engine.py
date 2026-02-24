"""
KB MCP Engine — Generates catalog prompt for LLM system prompt injection.

Loads the dependency registry and formats a compact catalog string
that can be injected into any LLM system prompt. The descriptions
act as triggers for the LLM to decide which tools to call.
"""

import json
from pathlib import Path
from typing import Optional


def generate_catalog_prompt(
    registry_path: Optional[Path] = None,
    filter_type: Optional[str] = None,
) -> str:
    """
    Generate a formatted catalog for injection into an LLM system prompt.

    Args:
        registry_path: Path to dependency_registry.json.
                       Defaults to the project root.
        filter_type: Optional type filter.

    Returns:
        Formatted markdown string listing available knowledge bases.
    """
    if registry_path is None:
        registry_path = Path(__file__).parent.parent / "dependency_registry.json"

    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    files = registry.get("files", {})

    lines = [
        "### AVAILABLE KNOWLEDGE BASE",
        "",
        "You have access to the following knowledge base documents. "
        "Use the `read` tool with the path to retrieve full content "
        "(including all dependencies).",
        "",
    ]

    for path in sorted(files.keys()):
        info = files[path]
        file_type = info.get("type") or "unknown"

        if filter_type and file_type != filter_type:
            continue

        basename = Path(path).name
        description = info.get("description") or "No description available."

        lines.append(f"- **{basename}** ({file_type}): {description}")

    lines.append("")
    lines.append(
        "IMPORTANT: Do NOT ask the user if you should look something up. "
        "If a question relates to any of the above topics, just use the "
        "`read` tool to fetch the relevant document."
    )

    return "\n".join(lines)


if __name__ == "__main__":
    # Quick preview of the generated catalog
    print(generate_catalog_prompt())
