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
- **Browse**: Filter knowledge by category (Skills, Logs, Plans).
- **Build**: Select the modules you need.
- **Inject**: Get a perfectly formatted, dependency-resolved text block ready to paste into your LLM.

## Directory Structure
- `src/`: Application source code (`app.py`, `dependency_manager.py`, `md_parser.py`).
- `content/`: The actual knowledge base files.
  - `agents/`: AI Agent definitions.
  - `plans/`: Project plans and roadmaps.
  - `guidelines/`: Core protocols and conventions.
- `dependency_registry.json`: The source of truth for file relationships.

## Usage
To run the Injector App locally:

```bash
pip install -r requirements.txt
streamlit run src/app.py
```
