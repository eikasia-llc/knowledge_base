#!/usr/bin/env python3
"""
Dependency Manager for Markdown Files

This module provides tools for:
1. Scanning MD files and extracting context_dependencies
2. Building/updating a structured dependency registry
3. Resolving dependencies recursively (depth-first)
4. Generating prompt injection text for AI assistants
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import date
from typing import Dict, List, Set, Optional, Tuple

# Local import
try:
    from .md_parser import MarkdownParser
except ImportError:
    from md_parser import MarkdownParser

SCRIPT_DIR = Path(__file__).parent.resolve()
# In the standalone repo, the project root is the parent of src/
PROJECT_ROOT = SCRIPT_DIR.parent


class DependencyManager:
    """Manages MD file dependencies for the Central Planner project."""

    # Registry is in the repo root, one level up from src/
    REGISTRY_FILE = SCRIPT_DIR.parent / "dependency_registry.json"

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = Path(project_root) if project_root else PROJECT_ROOT
        self.parser = MarkdownParser()
        self.registry = self._load_registry()

    def _load_registry(self) -> dict:
        """Load the dependency registry from JSON file."""
        if self.REGISTRY_FILE.exists():
            with open(self.REGISTRY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "version": "1.0.0",
            "description": "Structured registry of Markdown file dependencies",
            "last_updated": str(date.today()),
            "files": {}
        }

    def _save_registry(self):
        """Save the dependency registry to JSON file."""
        self.registry["last_updated"] = str(date.today())
        with open(self.REGISTRY_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, indent=2)
        print(f"Registry saved to: {self.REGISTRY_FILE}")

    def _get_relative_path(self, file_path: Path) -> str:
        """Get path relative to project root."""
        try:
            return str(file_path.resolve().relative_to(self.project_root.resolve()))
        except ValueError:
            return str(file_path)

    def _resolve_dependency_path(self, base_file: str, dep_path: str) -> str:
        """Resolve a dependency path relative to the base file."""
        base_dir = Path(base_file).parent
        resolved = (self.project_root / base_dir / dep_path).resolve()
        try:
            return str(resolved.relative_to(self.project_root.resolve()))
        except ValueError:
            return str(resolved)

    def extract_dependencies_from_file(self, file_path: Path) -> Dict[str, str]:
        """Extract context_dependencies from a single MD file."""
        try:
            root_node = self.parser.parse_file(str(file_path))

            # Check root node's metadata
            deps = root_node.metadata.get('context_dependencies', {})
            if isinstance(deps, dict):
                return deps
            return {}
        except Exception as e:
            print(f"Warning: Could not parse {file_path}: {e}", file=sys.stderr)
            return {}

    def scan_project(self, patterns: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Scan the project for MD files and extract their dependencies.

        Args:
            patterns: Optional list of glob patterns to match (default: **/*.md)

        Returns:
            Dictionary of file paths to their dependency info
        """
        if patterns is None:
            patterns = ["**/*.md"]

        scanned = {}

        for pattern in patterns:
            for md_file in self.project_root.glob(pattern):
                # Skip hidden directories and files
                if any(part.startswith('.') for part in md_file.parts):
                    continue

                rel_path = self._get_relative_path(md_file)
                deps = self.extract_dependencies_from_file(md_file)

                scanned[rel_path] = {
                    "path": rel_path,
                    "dependencies": deps
                }

        return scanned

    def update_registry(self, scanned_files: Optional[Dict] = None):
        """Update the registry with scanned files."""
        if scanned_files is None:
            scanned_files = self.scan_project()

        # Update files section
        for rel_path, info in scanned_files.items():
            if info["dependencies"]:  # Only add files with dependencies
                self.registry["files"][rel_path] = info

        self._save_registry()
        return self.registry

    def add_dependency(self, file_path: str, alias: str, dep_path: str):
        """
        Manually add a dependency to a file in the registry.

        Args:
            file_path: The file that has the dependency
            alias: Semantic alias for the dependency (e.g., 'conventions')
            dep_path: Path to the dependency file
        """
        if file_path not in self.registry["files"]:
            self.registry["files"][file_path] = {
                "path": file_path,
                "dependencies": {}
            }

        self.registry["files"][file_path]["dependencies"][alias] = dep_path
        self._save_registry()

    def remove_dependency(self, file_path: str, alias: str):
        """Remove a dependency from a file in the registry."""
        if file_path in self.registry["files"]:
            deps = self.registry["files"][file_path].get("dependencies", {})
            if alias in deps:
                del deps[alias]
                self._save_registry()

    def resolve_dependencies(
        self,
        file_path: str,
        _visited: Optional[Set[str]] = None,
        _result: Optional[List[str]] = None
    ) -> List[str]:
        """
        Recursively resolve all dependencies for a file (depth-first).

        The resolution follows MD_CONVENTIONS.md protocol:
        - Depth-First: If File A depends on B, and B depends on C,
          the agent reads C, then B, then A.

        Args:
            file_path: The target file to resolve dependencies for
            _visited: Internal - set of files currently in recursion stack (cycle detection)
            _result: Internal - accumulated result list

        Returns:
            Ordered list of file paths to read (dependencies first, target last)
        """
        # Initialize on first call
        if _visited is None:
            _visited = set()
        if _result is None:
            _result = []

        # Normalize the file path
        normalized = self._normalize_path(file_path)

        # Already processed - skip
        if normalized in _result:
            return _result

        # Cycle detection - currently processing this file
        if normalized in _visited:
            return _result

        _visited.add(normalized)

        # Get dependencies from registry
        file_info = self.registry["files"].get(normalized, {})
        deps = file_info.get("dependencies", {})

        # Recursively resolve each dependency (depth-first)
        for alias, dep_path in deps.items():
            resolved_path = self._resolve_dependency_path(normalized, dep_path)
            self.resolve_dependencies(resolved_path, _visited, _result)

        # Add the target file last (after its dependencies)
        if normalized not in _result:
            _result.append(normalized)

        _visited.discard(normalized)  # Allow revisiting from other paths

        return _result

    def _normalize_path(self, file_path: str) -> str:
        """Normalize a file path for consistent registry lookup."""
        # Handle relative paths like ../../MD_CONVENTIONS.md
        if file_path.startswith("../") or file_path.startswith("./"):
            # Try to resolve it from various base locations
            for base_file in self.registry["files"].keys():
                resolved = self._resolve_dependency_path(base_file, file_path)
                if resolved in self.registry["files"]:
                    return resolved

        # Try direct lookup
        if file_path in self.registry["files"]:
            return file_path

        # Try with/without leading paths
        for registered_path in self.registry["files"].keys():
            if registered_path.endswith(file_path) or file_path.endswith(registered_path):
                return registered_path

        return file_path

    def generate_prompt(
        self,
        file_path: str,
        format_type: str = "instruction"
    ) -> str:
        """
        Generate prompt injection text for reading a file with all its dependencies.

        Args:
            file_path: The target file (e.g., 'manager/cleaner/CLEANER_AGENT.md')
            format_type: Output format - 'instruction', 'list', or 'xml'

        Returns:
            Formatted prompt text
        """
        resolved = self.resolve_dependencies(file_path)

        if not resolved:
            return f"Read {file_path}"

        if format_type == "list":
            return "\n".join(f"- {f}" for f in resolved)

        elif format_type == "xml":
            lines = ["<required_readings>"]
            for i, f in enumerate(resolved, 1):
                is_target = (f == resolved[-1])
                tag = "target" if is_target else "dependency"
                lines.append(f'  <{tag} order="{i}">{f}</{tag}>')
            lines.append("</required_readings>")
            return "\n".join(lines)

        else:  # instruction format (default)
            if len(resolved) == 1:
                return f"Read {resolved[0]}"

            # Build instruction with dependencies explained
            dep_files = resolved[:-1]
            target_file = resolved[-1]

            lines = [
                f"To properly understand {target_file}, you must first read its dependencies in order:",
                ""
            ]

            for i, dep in enumerate(dep_files, 1):
                lines.append(f"{i}. Read {dep}")

            lines.append(f"{len(resolved)}. Finally, read {target_file}")
            lines.append("")
            lines.append("This ensures depth-first dependency resolution as per MD_CONVENTIONS.md.")

            return "\n".join(lines)

    def list_all_files(self) -> List[str]:
        """List all files in the registry."""
        return list(self.registry["files"].keys())

    def get_dependents(self, file_path: str) -> List[str]:
        """Find all files that depend on the given file."""
        normalized = self._normalize_path(file_path)
        dependents = []

        for reg_path, info in self.registry["files"].items():
            deps = info.get("dependencies", {})
            for alias, dep_path in deps.items():
                resolved = self._resolve_dependency_path(reg_path, dep_path)
                if resolved == normalized or dep_path.endswith(file_path):
                    dependents.append(reg_path)
                    break

        return dependents


def main():
    parser = argparse.ArgumentParser(
        description="Manage Markdown file dependencies and generate prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan project and update registry
  python dependency_manager.py scan

  # Resolve dependencies for an agent file
  python dependency_manager.py resolve manager/cleaner/CLEANER_AGENT.md

  # Generate prompt instruction for reading an agent
  python dependency_manager.py prompt manager/cleaner/CLEANER_AGENT.md

  # Generate XML format prompt
  python dependency_manager.py prompt manager/cleaner/CLEANER_AGENT.md --format xml

  # Add a dependency manually
  python dependency_manager.py add my_file.md --alias textbook --path docs/guide.md

  # List all registered files
  python dependency_manager.py list

  # Find files that depend on a given file
  python dependency_manager.py dependents MD_CONVENTIONS.md
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # scan command
    scan_parser = subparsers.add_parser("scan", help="Scan project and update registry")
    scan_parser.add_argument("--patterns", nargs="+", default=["**/*.md"],
                            help="Glob patterns to match (default: **/*.md)")

    # resolve command
    resolve_parser = subparsers.add_parser("resolve", help="Resolve dependencies for a file")
    resolve_parser.add_argument("file", help="Target file path")

    # prompt command
    prompt_parser = subparsers.add_parser("prompt", help="Generate prompt injection text")
    prompt_parser.add_argument("file", help="Target file path")
    prompt_parser.add_argument("--format", "-f", choices=["instruction", "list", "xml"],
                              default="instruction", help="Output format")

    # add command
    add_parser = subparsers.add_parser("add", help="Add a dependency to a file")
    add_parser.add_argument("file", help="File that has the dependency")
    add_parser.add_argument("--alias", "-a", required=True, help="Semantic alias")
    add_parser.add_argument("--path", "-p", required=True, help="Dependency path")

    # remove command
    remove_parser = subparsers.add_parser("remove", help="Remove a dependency from a file")
    remove_parser.add_argument("file", help="File to remove dependency from")
    remove_parser.add_argument("--alias", "-a", required=True, help="Alias to remove")

    # list command
    subparsers.add_parser("list", help="List all registered files")

    # dependents command
    dep_parser = subparsers.add_parser("dependents", help="Find files that depend on a file")
    dep_parser.add_argument("file", help="Target file path")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    manager = DependencyManager()

    if args.command == "scan":
        print("Scanning project for MD files...")
        scanned = manager.scan_project(args.patterns)
        print(f"Found {len(scanned)} MD files")
        manager.update_registry(scanned)

        # Print summary of files with dependencies
        with_deps = {k: v for k, v in scanned.items() if v["dependencies"]}
        print(f"\nFiles with dependencies: {len(with_deps)}")
        for path, info in with_deps.items():
            deps = info["dependencies"]
            print(f"  {path}: {list(deps.keys())}")

    elif args.command == "resolve":
        resolved = manager.resolve_dependencies(args.file)
        print(f"Dependency resolution order for {args.file}:")
        for i, f in enumerate(resolved, 1):
            marker = " <- target" if i == len(resolved) else ""
            print(f"  {i}. {f}{marker}")

    elif args.command == "prompt":
        prompt = manager.generate_prompt(args.file, args.format)
        print(prompt)

    elif args.command == "add":
        manager.add_dependency(args.file, args.alias, args.path)
        print(f"Added dependency: {args.file} -> {args.alias}: {args.path}")

    elif args.command == "remove":
        manager.remove_dependency(args.file, args.alias)
        print(f"Removed dependency '{args.alias}' from {args.file}")

    elif args.command == "list":
        files = manager.list_all_files()
        print(f"Registered files ({len(files)}):")
        for f in sorted(files):
            print(f"  {f}")

    elif args.command == "dependents":
        dependents = manager.get_dependents(args.file)
        if dependents:
            print(f"Files that depend on {args.file}:")
            for f in dependents:
                print(f"  {f}")
        else:
            print(f"No files depend on {args.file}")


if __name__ == "__main__":
    main()
