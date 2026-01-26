# Prompting Module
- status: active
- type: guideline
- context_dependencies: {"conventions": "../../MD_CONVENTIONS.md"}
<!-- content -->
This module provides tools for managing Markdown file dependencies and generating prompt injection text for AI coding assistants.

## Purpose
- status: active
<!-- content -->
1. **Structured Dependency Tracking**: Replace inline `context_dependencies` annotations with a centralized registry
2. **Automatic Dependency Resolution**: When you say "Read CLEANER_AGENT.md", automatically include all dependencies
3. **Prompt Generation**: Generate ready-to-use prompts that force AI assistants to read files in the correct order

## Quick Start
- status: active
<!-- content -->

### Generate a Prompt
```bash
cd manager/prompting
python prompt_generator.py CLEANER_AGENT.md
```

Output:
```
Read these files in order:
1. README.md
2. MD_CONVENTIONS.md
3. AGENTS.md
4. manager/cleaner/CLEANER_AGENT.md (target)
```

### Scan and Update Registry
```bash
python dependency_manager.py scan
```

## Files
- status: active
<!-- content -->
| File | Description |
|------|-------------|
| `dependency_registry.json` | Centralized JSON registry of all MD file dependencies |
| `dependency_manager.py` | Core module for scanning, resolving, and managing dependencies |
| `prompt_generator.py` | User-friendly CLI for generating AI prompts |

## dependency_manager.py
- status: active
- type: context
<!-- content -->
The main module for dependency management.

### Commands

#### scan
Scan the project for MD files and update the registry:
```bash
python dependency_manager.py scan
python dependency_manager.py scan --patterns "AI_AGENTS/**/*.md" "manager/**/*.md"
```

#### resolve
Show the dependency resolution order for a file:
```bash
python dependency_manager.py resolve manager/cleaner/CLEANER_AGENT.md
```

#### prompt
Generate prompt injection text:
```bash
python dependency_manager.py prompt CLEANER_AGENT.md
python dependency_manager.py prompt CLEANER_AGENT.md --format xml
python dependency_manager.py prompt CLEANER_AGENT.md --format list
```

#### add/remove
Manually manage dependencies:
```bash
python dependency_manager.py add my_agent.md --alias textbook --path docs/guide.md
python dependency_manager.py remove my_agent.md --alias textbook
```

#### list
List all registered files:
```bash
python dependency_manager.py list
```

#### dependents
Find files that depend on a given file:
```bash
python dependency_manager.py dependents MD_CONVENTIONS.md
```

## prompt_generator.py
- status: active
- type: context
<!-- content -->
A streamlined tool for generating AI prompts.

### Basic Usage
```bash

# Simple prompt generation
python prompt_generator.py CLEANER_AGENT.md

# Verbose mode (shows resolution process)
python prompt_generator.py CLEANER_AGENT.md --verbose

# Copy to clipboard (macOS)
python prompt_generator.py CLEANER_AGENT.md --copy
```

### Prompt Styles
**Imperative (default)**:
```
Read these files in order:
1. README.md
2. MD_CONVENTIONS.md
3. AGENTS.md
4. manager/cleaner/CLEANER_AGENT.md (target)
```

**Polite**:
```bash
python prompt_generator.py CLEANER_AGENT.md --style polite
```
```
Please read the following files in order to understand CLEANER_AGENT.md:

1. README.md
2. MD_CONVENTIONS.md
3. AGENTS.md
4. manager/cleaner/CLEANER_AGENT.md
```

**Structured (XML)**:
```bash
python prompt_generator.py CLEANER_AGENT.md --style structured
```
```xml
<!-- DEPENDENCY INJECTION: Read these files in order -->
<dependency_chain>
  <file order="1" role="dependency">README.md</file>
  <file order="2" role="dependency">MD_CONVENTIONS.md</file>
  <file order="3" role="dependency">AGENTS.md</file>
  <file order="4" role="target">manager/cleaner/CLEANER_AGENT.md</file>
</dependency_chain>
```

### With Rationale
```bash
python prompt_generator.py CLEANER_AGENT.md --rationale
```
Adds explanation of why dependencies are needed.

## dependency_registry.json
- status: active
- type: context
<!-- content -->
The registry is a JSON file that maps MD files to their dependencies:

```json
{
  "version": "1.0.0",
  "last_updated": "2026-01-25",
  "files": {
    "manager/cleaner/CLEANER_AGENT.md": {
      "path": "manager/cleaner/CLEANER_AGENT.md",
      "dependencies": {
        "conventions": "MD_CONVENTIONS.md",
        "agents": "AGENTS.md"
      }
    }
  }
}
```

### Benefits Over Inline Annotations
| Inline (`context_dependencies`) | Registry (`dependency_registry.json`) |
|--------------------------------|--------------------------------------|
| Scattered across files | Centralized, single source of truth |
| Relative paths only | Normalized to project root |
| Manual resolution | Automatic recursive resolution |
| Hard to query | Easy to query and analyze |

## Workflow
- status: active
- type: protocol
<!-- content -->

### For Users
1. **Before talking to an AI assistant**:
   ```bash
   python prompt_generator.py AGENT_NAME.md --copy
   ```

2. **Paste the output** as your first message to the AI

3. **The AI reads all dependencies** in the correct order before reading your target file

### For Developers
1. **When adding a new agent/file**:
   ```bash
   python dependency_manager.py add my_new_agent.md --alias conventions --path MD_CONVENTIONS.md
   python dependency_manager.py add my_new_agent.md --alias agents --path AGENTS.md
   ```

2. **Periodically scan** to sync with `context_dependencies` in files:
   ```bash
   python dependency_manager.py scan
   ```

## Resolution Algorithm
- status: active
- type: context
<!-- content -->
Dependencies are resolved **depth-first**, following the protocol in `MD_CONVENTIONS.md`:

> If File A depends on B, and B depends on C, the agent reads C, then B, then A.

Example:
- `CLEANER_AGENT.md` depends on `AGENTS.md` and `MD_CONVENTIONS.md`
- `AGENTS.md` depends on `MD_CONVENTIONS.md` and `README.md`
- `MD_CONVENTIONS.md` depends on `README.md`

Resolution order:
1. `README.md` (leaf dependency)
2. `MD_CONVENTIONS.md` (depends only on README)
3. `AGENTS.md` (depends on above)
4. `CLEANER_AGENT.md` (target)
