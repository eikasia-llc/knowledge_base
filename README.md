# Knowledge Base Repository
- status: active
- context_dependencies: {}
<!-- content -->

## Overview
This repository serves as the central nervous system for **Prompt Context Management**. It tracks, organizes, and serves Markdown-based knowledge modules (Agents, Plans, Guidelines, etc.) that are used to "inject" context into Large Language Model (LLM) prompts.

## Key Features

### 1. Context Tracking & Storage
- **Centralized Knowledge**: Stores critical project documentation in `content/`, organized by type (Agents, Guidelines, Protocols).
- **Dependency Resolution**: Implements a strict **Depth-First Dependency Resolution** protocol. If you need `MC_AGENT.md`, the system automatically ensures you also get `AGENTS.md` and `MD_CONVENTIONS.md` in the correct order.
- **Dependency Registry**: Maintains a `dependency_registry.json` that maps every file to its required dependencies, ensuring no context is missing.

### 2. Prompt Injection Logic
- **Automated assembly**: Logic in `src/dependency_manager.py` resolves the dependency graph for any given file and generates a linear reading list.
- **Protocol Compliance**: Ensures that the resulting context block follows the project's strict Markdown-JSON Hybrid Schema.

### 3. The Knowledge Base Injector App
A built-in Streamlit application allows users to visually browse and assemble prompts.

🚀 **[Launch Active App](https://knowledgebases-eikasia.streamlit.app/)**
- **Browse**: Filter knowledge by category (Skills, Logs, Plans).
- **Build**: Select the modules you need.
- **Inject**: Get a perfectly formatted, dependency-resolved text block ready to paste into your LLM.

## Directory Structure
- `src/`: Application source code (`app.py`, `dependency_manager.py`, `md_parser.py`).
- `content/`: The actual knowledge base files.
  - `agents/`: AI Agent definitions.
  - `plans/`: Project plans and roadmaps.
  - `core/`: Core conventions, agents, and protocols.
  - `guidelines/`: Additional guidelines and templates.
  - `logs/`: Operational logs and artifacts.
- `manager/`: Maintenance and management tools.
  - `cleaner/`: Pipeline for ingesting and cleaning external repositories.
  - `language/`: Tools for Markdown parsing and schema enforcement.
- `dependency_registry.json`: The source of truth for file relationships.

## Maintenance & Cleaning Protocol
To update the knowledge base with content from external repositories, follow this strict protocol:

1.  **Update Repository List**: Add the target repository URLs to `manager/cleaner/toclean_repolist.txt`.
2.  **Run Pipeline**: Execute the cleaning pipeline:
    ```bash
    python3 manager/cleaner/pipeline.py
    ```
    This script will clone the repositories, migrate Markdown files to the correct schema, and save them to `manager/cleaner/temprepo_cleaning/`.
3.  **Integrate Content**:
    -   Review files in `manager/cleaner/temprepo_cleaning/`.
    -   **Action**: Move *only* new files into the appropriate `content/` subdirectories (`agents/`, `plans/`, `core/`, etc.).
    -   **Constraint**: Do NOT overwrite existing files in `content/` unless explicitly instructed. Current working files must be preserved.

## Usage
To run the Injector App locally:

```bash
pip install -r requirements.txt
streamlit run src/app.py
```
