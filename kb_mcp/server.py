"""
KB MCP Server — FastMCP server exposing knowledge base tools.

Follows the project's MCP convention (see MCP_SKILL.md):
  - Tools are defined in tools.py (pure logic)
  - This file registers them with FastMCP and adds rich schemas
  - Run via: python -m kb_mcp.server  OR  python -m kb_mcp
"""

import logging
from typing import Optional, List

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from .tools import list_knowledge_catalog, read_knowledge, search_knowledge

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kb_mcp_server")

# Initialize FastMCP
mcp = FastMCP("kb_mcp")


# ---------------------------------------------------------------------------
# Tool registrations
# ---------------------------------------------------------------------------

@mcp.tool()
async def catalog(
    filter_type: Optional[str] = Field(
        None,
        description=(
            "Optional type filter. Valid values: "
            "agent_skill, guideline, plan, log, documentation, task. "
            "Omit to list all entries."
        ),
    ),
) -> dict:
    """List all available knowledge base documents with their descriptions.

    Use this tool FIRST to discover what knowledge is available.
    Each entry has a path, type, and short description so you can decide
    which documents to read in full.
    """
    logger.info(f"catalog called (filter_type={filter_type})")
    return list_knowledge_catalog(filter_type=filter_type)


@mcp.tool()
async def read(
    path: str = Field(
        ...,
        description=(
            "Registry path of the target file, e.g. "
            "'content/skills/MCP_SKILL.md'. "
            "Use the 'catalog' tool to discover valid paths."
        ),
    ),
) -> dict:
    """Read a knowledge base document and ALL its dependencies.

    This resolves the full dependency chain (depth-first) and returns
    the content of every file the target depends on, followed by the
    target itself. This ensures you have the complete context.
    """
    logger.info(f"read called (path={path})")
    return read_knowledge(path)


@mcp.tool()
async def search(
    query: str = Field(
        ...,
        description="Search keywords (case-insensitive substring match).",
    ),
    max_results: int = Field(
        10,
        description="Maximum results to return (default: 10).",
    ),
) -> dict:
    """Search knowledge base descriptions and paths by keyword.

    Use this when you know a topic but not the exact file path.
    Returns matching entries with their descriptions.
    """
    logger.info(f"search called (query={query!r})")
    return search_knowledge(query, max_results=max_results)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run():
    """Start the MCP server."""
    logger.info("Starting KB MCP Server...")
    mcp.run()


if __name__ == "__main__":
    run()
