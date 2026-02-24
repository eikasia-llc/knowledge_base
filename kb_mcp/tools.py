"""
KB MCP Tools — Pure logic functions for knowledge base access.

These functions operate on dependency_registry.json and the markdown files.
No MCP dependency here — server.py wraps these as MCP tools.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import sys

# Add src/ to path so we can import DependencyManager
_SRC_DIR = str(Path(__file__).parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from dependency_manager import DependencyManager


# ---------------------------------------------------------------------------
# Globals (lazy-loaded)
# ---------------------------------------------------------------------------
_dep_manager: Optional[DependencyManager] = None


def _get_manager() -> DependencyManager:
    global _dep_manager
    if _dep_manager is None:
        project_root = Path(__file__).parent.parent
        _dep_manager = DependencyManager(project_root=project_root)
    return _dep_manager


def _load_registry() -> Dict:
    """Load the dependency registry JSON."""
    manager = _get_manager()
    with open(manager.REGISTRY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Tool 1: List Knowledge Catalog
# ---------------------------------------------------------------------------
def list_knowledge_catalog(filter_type: Optional[str] = None) -> Dict:
    """
    Return a compact catalog of all knowledge base entries.

    Each entry includes path, type, and description — enough for the LLM
    to decide what to read in detail.

    Args:
        filter_type: Optional type filter (e.g. 'agent_skill', 'guideline',
                     'plan', 'log', 'documentation', 'task').

    Returns:
        Dict with 'catalog' (list of entries) and 'total' count.
    """
    registry = _load_registry()
    files = registry.get("files", {})

    catalog = []
    for path, info in sorted(files.items()):
        file_type = info.get("type")
        if filter_type and file_type != filter_type:
            continue
        catalog.append({
            "path": path,
            "type": file_type,
            "description": info.get("description"),
        })

    return {"catalog": catalog, "total": len(catalog)}


# ---------------------------------------------------------------------------
# Tool 2: Read Knowledge (with dependency resolution)
# ---------------------------------------------------------------------------
def read_knowledge(path: str) -> Dict:
    """
    Read a knowledge base file and all its resolved dependencies.

    Uses DependencyManager.resolve_dependencies() to produce a depth-first
    reading order, then reads each file's full content from disk.

    Args:
        path: Registry path of the target file
              (e.g. 'content/skills/MCP_SKILL.md').

    Returns:
        Dict with 'requested_file', 'dependency_chain' (ordered list),
        and 'contents' (list of {path, content} objects).
    """
    manager = _get_manager()

    # Resolve full dependency chain (depth-first: deps first, target last)
    chain = manager.resolve_dependencies(path)

    contents = []
    for file_path in chain:
        abs_path = manager.project_root / file_path
        try:
            text = abs_path.read_text(encoding="utf-8")
        except Exception as e:
            text = f"[Error reading file: {e}]"
        contents.append({"path": file_path, "content": text})

    return {
        "requested_file": path,
        "dependency_chain": chain,
        "contents": contents,
    }


# ---------------------------------------------------------------------------
# Tool 3: Search Knowledge
# ---------------------------------------------------------------------------
def search_knowledge(query: str, max_results: int = 10) -> Dict:
    """
    Simple keyword search across knowledge base descriptions.

    Searches the 'description' and 'path' fields in the registry for
    case-insensitive substring matches.

    Args:
        query: Search term(s).
        max_results: Maximum number of results to return.

    Returns:
        Dict with 'results' (list of matching entries) and 'total' count.
    """
    registry = _load_registry()
    files = registry.get("files", {})
    query_lower = query.lower()

    results = []
    for path, info in sorted(files.items()):
        description = info.get("description") or ""
        # Search in path and description
        if query_lower in path.lower() or query_lower in description.lower():
            results.append({
                "path": path,
                "type": info.get("type"),
                "description": description,
            })
            if len(results) >= max_results:
                break

    return {"results": results, "total": len(results)}
