# Knowledge Base Repository
- status: active
- type: guideline
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
- **Inject**: Download a ZIP bundle (`context_bundle.zip`) containing all required markdown files, ready for upload to your LLM workspace.

## Directory Structure
- `src/`: Application source code (`app.py`, `dependency_manager.py`, `md_parser.py`).
- `content/`: The actual knowledge base files.
  - `agents/`: AI Agent definitions.
  - `plans/`: Project plans and roadmaps.
  - `guidelines/`: Additional guidelines and templates.
  - `logs/`: Operational logs and artifacts.
- `manager/`: Maintenance and management tools.
  - `cleaner/`: Pipeline for ingesting and cleaning external repositories.
  - `language/`: Tools for Markdown parsing and schema enforcement.
- `AGENTS.md`: Core agent protocols and workflow.
- `MD_CONVENTIONS.md`: The Markdown-JSON Hybrid Schema specification.
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

## Streamlit app

The main command for running a Streamlit app is `streamlit run`. Streamlit also offers several other command-line interface (CLI) tools for configuration, documentation, and example apps. 

`streamlit run your_app.py` starts a local web server and opens your app in a new tab in your default web browser, typically at http://localhost:8501. 
You can also pass a URL to streamlit run if your script is hosted remotely, such as on GitHub: 
streamlit run https://raw.githubusercontent.com 


### Running the Meta MCP Server
To enable dynamic context discovery in your IDE:

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements-mcp.txt
    ```
2.  **Build the Index** (First run only):
    ```bash
    python src/index_builder.py .
    ```
3.  **Configure your IDE** (e.g., `claude_desktop_config.json`):
    ```json
    {
      "mcpServers": {
        "meta-knowledge": {
          "command": "python",
          "args": ["-m", "src.mcp_server"],
          "cwd": "/path/to/knowledge_bases"
        }
      }
    }
    ```

## Deploying

### Streamlit app to GCloud Run

The `deploy.sh` handles that. Read it completely to understand what it needs.

#### Permissions

To access the app: Grant access to your specific account (Recommended)
Instead of making it public, we grant ourseles permission to view it. 
This is the command:

```
gcloud run services add-iam-policy-binding knowledge-base-app \
    --region=us-central1 \
    --member="user:eikasia@eikasia.com" \
    --role="roles/run.invoker" \
    --project=eikasia-ops
```

With just this we would see the 403 Forbidden error because Cloud Run private services (which this is, due to your Org Policy) require an Authorization header with a valid identity token in every request. A standard browser visit doesn't send this token automatically, even if you have the right permissions.

To solve this we use the Cloud Run Proxy. This creates a local bridge that handles the authentication for you.

This is the command:

```
gcloud run services proxy knowledge-base-app --region=us-central1 --project=eikasia-ops
```


gcloud run services add-iam-policy-binding

