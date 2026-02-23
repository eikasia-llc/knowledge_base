# Meta MCP Agent Protocol
- id: meta_mcp_agent
- status: active
- type: guideline
- owner: eikasia-llc
- last_checked: 2025-01-29
- label: ['protocol']
<!-- content -->
This document defines the **Meta MCP Agent Protocol**—a Model Context Protocol (MCP) server that enables coding assistants to dynamically discover, select, and retrieve relevant knowledge from structured knowledge bases.

The core insight: before an LLM can effectively perform a task, it must first understand *what context it needs*. This protocol provides that meta-cognitive layer.

## Problem Statement
- id: meta_mcp_agent.problem_statement
- status: active
- type: documentation
<!-- content -->
Coding assistants face a fundamental challenge: they cannot know what they don't know. When presented with a task, an LLM may lack critical context about project conventions, architectural decisions, domain knowledge, or existing patterns.

Current solutions suffer from two failure modes: (a) dumping all context into the prompt (expensive, noisy, hits token limits), or (b) requiring users to manually specify which documents to include (friction, requires expertise).

The Meta MCP Agent solves this by providing **semantic discovery**—the LLM asks "what do I need to know to do X?" and receives precisely the relevant knowledge.

## Architecture Overview
- id: meta_mcp_agent.architecture
- status: active
- type: documentation
<!-- content -->
```
┌─────────────────────────────────────────────────────────────────┐
│                        Client (IDE/CLI)                         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MCP Client (LLM Agent)                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  1. Receive task from user                                │  │
│  │  2. Call `discover_context` with task description         │  │
│  │  3. Receive ranked knowledge base recommendations         │  │
│  │  4. Call `retrieve_knowledge` for selected items          │  │
│  │  5. Execute task with full context                        │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Meta MCP Server                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Index     │  │  Semantic   │  │   Knowledge Base        │  │
│  │   Manager   │◄─┤  Matcher    │◄─┤   Reader                │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Knowledge Bases (Markdown)                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │guidelines│ │protocols │ │ context  │ │  skills  │  ...      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

## MCP Tools Specification
- id: meta_mcp_agent.tools
- status: active
- type: guideline
- label: ['protocol']
<!-- content -->
The Meta MCP Server exposes the following tools via the Model Context Protocol.

### Tool: discover_context
- id: meta_mcp_agent.tools.discover_context
- status: active
- type: guideline
- label: ['protocol']
<!-- content -->
Analyzes a task description and returns ranked recommendations of relevant knowledge bases.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "task_description": {
      "type": "string",
      "description": "Natural language description of the task the LLM is about to perform"
    },
    "task_type": {
      "type": "string",
      "enum": ["implement", "debug", "refactor", "document", "review", "design", "test"],
      "description": "Optional categorization to improve matching"
    },
    "current_file": {
      "type": "string",
      "description": "Optional path to the file currently being worked on"
    },
    "max_results": {
      "type": "integer",
      "default": 5,
      "description": "Maximum number of recommendations to return"
    }
  },
  "required": ["task_description"]
}
```

**Output Schema:**
```json
{
  "type": "object",
  "properties": {
    "recommendations": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": { "type": "string" },
          "path": { "type": "string" },
          "title": { "type": "string" },
          "type": { "type": "string" },
          "relevance_score": { "type": "number" },
          "reason": { "type": "string" },
          "estimated_tokens": { "type": "integer" }
        }
      }
    },
    "total_available": { "type": "integer" }
  }
}
```

**Example Call:**
```json
{
  "tool": "discover_context",
  "arguments": {
    "task_description": "Implement user authentication with JWT tokens",
    "task_type": "implement",
    "max_results": 3
  }
}
```

**Example Response:**
```json
{
  "recommendations": [
    {
      "id": "guidelines.security.authentication",
      "path": "guidelines/SECURITY.md#authentication",
      "title": "Authentication Guidelines",
      "type": "guideline",
      "relevance_score": 0.94,
      "reason": "Directly covers authentication patterns and JWT best practices",
      "estimated_tokens": 1200
    },
    {
      "id": "context.architecture.services",
      "path": "context/ARCHITECTURE.md#services",
      "title": "Service Architecture",
      "type": "context",
      "relevance_score": 0.78,
      "reason": "Defines where auth logic should live in the codebase",
      "estimated_tokens": 800
    }
  ],
  "total_available": 47
}
```

### Tool: retrieve_knowledge
- id: meta_mcp_agent.tools.retrieve_knowledge
- status: active
- type: guideline
- label: ['protocol']
<!-- content -->
Fetches the full content of one or more knowledge base nodes.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "ids": {
      "type": "array",
      "items": { "type": "string" },
      "description": "List of node IDs to retrieve"
    },
    "include_children": {
      "type": "boolean",
      "default": false,
      "description": "Whether to include child nodes in the response"
    },
    "format": {
      "type": "string",
      "enum": ["markdown", "json", "plain"],
      "default": "markdown"
    }
  },
  "required": ["ids"]
}
```

**Output Schema:**
```json
{
  "type": "object",
  "properties": {
    "nodes": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": { "type": "string" },
          "path": { "type": "string" },
          "title": { "type": "string" },
          "metadata": { "type": "object" },
          "content": { "type": "string" },
          "children": { "type": "array" }
        }
      }
    },
    "total_tokens": { "type": "integer" }
  }
}
```

### Tool: list_knowledge_bases
- id: meta_mcp_agent.tools.list_knowledge_bases
- status: active
- type: guideline
- label: ['protocol']
<!-- content -->
Returns a catalog of all available knowledge bases with their metadata.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "filter_type": {
      "type": "string",
      "enum": ["guideline", "protocol", "context", "agent_skill", "log", "plan", "task"],
      "description": "Optional filter by node type"
    },
    "filter_status": {
      "type": "string",
      "enum": ["active", "draft", "deprecated"],
      "description": "Optional filter by status"
    }
  }
}
```

**Output Schema:**
```json
{
  "type": "object",
  "properties": {
    "knowledge_bases": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": { "type": "string" },
          "path": { "type": "string" },
          "title": { "type": "string" },
          "type": { "type": "string" },
          "status": { "type": "string" },
          "description": { "type": "string" },
          "last_checked": { "type": "string" },
          "node_count": { "type": "integer" }
        }
      }
    }
  }
}
```

### Tool: search_knowledge
- id: meta_mcp_agent.tools.search_knowledge
- status: active
- type: guideline
- label: ['protocol']
<!-- content -->
Performs full-text and semantic search across all knowledge bases.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Search query (supports natural language)"
    },
    "search_type": {
      "type": "string",
      "enum": ["semantic", "keyword", "hybrid"],
      "default": "hybrid"
    },
    "scope": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Optional list of knowledge base IDs to search within"
    },
    "max_results": {
      "type": "integer",
      "default": 10
    }
  },
  "required": ["query"]
}
```

### Tool: get_dependencies
- id: meta_mcp_agent.tools.get_dependencies
- status: active
- type: guideline
- label: ['protocol']
<!-- content -->
Given a node ID, returns all nodes it depends on (via `blocked_by`) and all nodes that depend on it.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "node_id": {
      "type": "string",
      "description": "The node ID to analyze"
    },
    "direction": {
      "type": "string",
      "enum": ["upstream", "downstream", "both"],
      "default": "both"
    },
    "depth": {
      "type": "integer",
      "default": 1,
      "description": "How many levels of dependencies to traverse"
    }
  },
  "required": ["node_id"]
}
```

## MCP Resources Specification
- id: meta_mcp_agent.resources
- status: active
- type: guideline
- label: ['protocol']
<!-- content -->
The server exposes knowledge bases as MCP Resources for direct access.

### Resource: knowledge_base
- id: meta_mcp_agent.resources.knowledge_base
- status: active
- type: guideline
- label: ['protocol']
<!-- content -->
Each knowledge base file is exposed as a resource with URI pattern:

```
knowledge://{base_path}/{file_path}#{node_id}
```

**Examples:**
- `knowledge://eikasia/guidelines/SECURITY.md` — Full file
- `knowledge://eikasia/guidelines/SECURITY.md#authentication` — Specific section
- `knowledge://eikasia/context/ARCHITECTURE.md#services.api` — Nested node

**Resource Metadata:**
```json
{
  "uri": "knowledge://eikasia/guidelines/SECURITY.md#authentication",
  "name": "Authentication Guidelines",
  "mimeType": "text/markdown",
  "metadata": {
    "id": "guidelines.security.authentication",
    "type": "guideline",
    "status": "active",
    "last_checked": "2025-01-29"
  }
}
```

## MCP Prompts Specification
- id: meta_mcp_agent.prompts
- status: active
- type: guideline
- label: ['protocol']
<!-- content -->
Pre-defined prompts that guide LLMs in using the knowledge base effectively.

### Prompt: task_preparation
- id: meta_mcp_agent.prompts.task_preparation
- status: active
- type: guideline
- label: ['protocol']
<!-- content -->
Guides the LLM through context discovery before task execution.

```json
{
  "name": "task_preparation",
  "description": "Prepare context before executing a coding task",
  "arguments": [
    {
      "name": "task",
      "description": "The task to prepare for",
      "required": true
    }
  ]
}
```

**Prompt Template:**
```
You are about to perform the following task:
{task}

Before proceeding, follow these steps:

1. Call `discover_context` with the task description to identify relevant knowledge
2. Review the recommendations and their relevance scores
3. Call `retrieve_knowledge` for items with relevance_score > 0.7
4. Read and internalize the retrieved context
5. Only then proceed with the task, applying the guidelines and patterns you learned

If you encounter ambiguity during the task, call `search_knowledge` to find specific guidance.
```

### Prompt: knowledge_audit
- id: meta_mcp_agent.prompts.knowledge_audit
- status: active
- type: guideline
- label: ['protocol']
<!-- content -->
Helps the LLM verify it has sufficient context for a task.

```json
{
  "name": "knowledge_audit",
  "description": "Verify sufficient context exists before proceeding",
  "arguments": [
    {
      "name": "task",
      "required": true
    },
    {
      "name": "loaded_context",
      "description": "List of knowledge base IDs already loaded",
      "required": true
    }
  ]
}
```

## Implementation Guide
- id: meta_mcp_agent.implementation
- status: active
- type: plan
<!-- content -->
Step-by-step instructions for implementing the Meta MCP Server.

### Phase 1: Index Builder
- id: meta_mcp_agent.implementation.index_builder
- status: todo
- type: task
- estimate: 2d
- blocked_by: []
<!-- content -->
Build the knowledge base index that powers discovery.

**Requirements:**
1. Parse all `.md` files using the Markdown-JSON hybrid schema parser (`language/md_parser.py`)
2. Extract metadata from each node (id, type, status, title)
3. Generate embeddings for semantic search (use `sentence-transformers` or OpenAI embeddings)
4. Build an inverted index for keyword search
5. Store the index in a format that supports fast updates (SQLite + FAISS recommended)

**Index Schema:**
```python
@dataclass
class IndexEntry:
    id: str                    # Unique node ID
    path: str                  # File path + anchor
    title: str                 # Node title (header text)
    type: str                  # Node type from metadata
    status: str                # Node status
    content_hash: str          # For change detection
    embedding: List[float]     # Semantic vector
    keywords: List[str]        # Extracted keywords
    parent_id: Optional[str]   # For hierarchy traversal
    children_ids: List[str]    # Direct children
    blocked_by: List[str]      # Explicit dependencies
    token_count: int           # Estimated tokens
```

**File:** `src/index_builder.py`

### Phase 2: Semantic Matcher
- id: meta_mcp_agent.implementation.semantic_matcher
- status: todo
- type: task
- estimate: 2d
- blocked_by: [meta_mcp_agent.implementation.index_builder]
<!-- content -->
Implement the matching logic for `discover_context`.

**Algorithm:**
1. Generate embedding for task description
2. Perform approximate nearest neighbor search against index
3. Apply boost factors based on:
   - Node type relevance to task_type (e.g., `guideline` boosts for `implement`)
   - Recency (newer `last_checked` dates rank higher)
   - Status (active > draft > deprecated)
   - File proximity if `current_file` is provided
4. De-duplicate overlapping nodes (prefer parent if children are also matched)
5. Return top-k with explanations

**Boost Matrix (task_type → node_type):**
| task_type   | guideline | protocol | context | agent_skill |
|-------------|-----------|----------|---------|-------------|
| implement   | 1.5       | 1.2      | 1.3     | 1.0         |
| debug       | 1.0       | 1.3      | 1.5     | 0.8         |
| refactor    | 1.3       | 1.1      | 1.4     | 0.9         |
| document    | 1.2       | 1.0      | 1.5     | 0.7         |
| review      | 1.4       | 1.5      | 1.2     | 0.8         |
| design      | 1.3       | 1.4      | 1.5     | 1.1         |
| test        | 1.2       | 1.3      | 1.2     | 1.0         |

**File:** `src/semantic_matcher.py`

### Phase 3: MCP Server Core
- id: meta_mcp_agent.implementation.server_core
- status: todo
- type: task
- estimate: 3d
- blocked_by: [meta_mcp_agent.implementation.semantic_matcher]
<!-- content -->
Implement the MCP server using the official Python SDK.

**Dependencies:**
```
mcp>=1.0.0
uvicorn
fastapi  # Optional, for HTTP transport
```

**Server Structure:**
```python
from mcp.server import Server
from mcp.types import Tool, Resource, Prompt

server = Server("meta-knowledge-mcp")

@server.tool("discover_context")
async def discover_context(task_description: str, ...) -> dict:
    ...

@server.tool("retrieve_knowledge")
async def retrieve_knowledge(ids: list[str], ...) -> dict:
    ...

@server.resource("knowledge://{path}")
async def get_knowledge_resource(path: str) -> Resource:
    ...

@server.prompt("task_preparation")
async def task_preparation_prompt(task: str) -> str:
    ...
```

**File:** `src/mcp_server.py`

### Phase 4: Index Synchronization
- id: meta_mcp_agent.implementation.sync
- status: todo
- type: task
- estimate: 1d
- blocked_by: [meta_mcp_agent.implementation.index_builder]
<!-- content -->
Implement efficient index updates when knowledge bases change.

**Strategies:**
1. **File watcher**: Use `watchdog` to detect `.md` file changes
2. **Hash-based diff**: Only re-index nodes whose content hash changed
3. **Incremental embedding**: Update only affected embeddings
4. **Git hook integration**: Trigger re-index on commit

**File:** `src/index_sync.py`

### Phase 5: Client Integration Examples
- id: meta_mcp_agent.implementation.client_examples
- status: todo
- type: task
- estimate: 1d
- blocked_by: [meta_mcp_agent.implementation.server_core]
<!-- content -->
Provide integration examples for popular coding assistants.

**Claude Desktop (`claude_desktop_config.json`):**
```json
{
  "mcpServers": {
    "meta-knowledge": {
      "command": "python",
      "args": ["-m", "meta_mcp_agent.server"],
      "env": {
        "KNOWLEDGE_BASE_PATH": "/path/to/knowledge_bases"
      }
    }
  }
}
```

**VS Code Continue:**
```json
{
  "models": [...],
  "mcpServers": [
    {
      "name": "meta-knowledge",
      "transport": "stdio",
      "command": "python -m meta_mcp_agent.server"
    }
  ]
}
```

**Cursor (`.cursor/mcp.json`):**
```json
{
  "servers": {
    "meta-knowledge": {
      "command": "python",
      "args": ["-m", "meta_mcp_agent.server"],
      "cwd": "/path/to/knowledge_bases"
    }
  }
}
```

## Agent Behavior Guidelines
- id: meta_mcp_agent.agent_behavior
- status: active
- type: agent_skill
<!-- content -->
Instructions for LLMs on how to effectively use this protocol.

### Discovery-First Principle
- id: meta_mcp_agent.agent_behavior.discovery_first
- status: active
- type: guideline
<!-- content -->
**Always call `discover_context` before starting a non-trivial task.**

Before writing code, modifying files, or making architectural decisions, the agent MUST:

1. Formulate a clear task description
2. Call `discover_context` with appropriate `task_type`
3. Review recommendations with `relevance_score > 0.6`
4. Load high-relevance knowledge via `retrieve_knowledge`
5. Proceed only after internalizing relevant guidelines

**Exception:** Trivial tasks (typo fixes, simple renames) may skip discovery.

### Relevance Thresholds
- id: meta_mcp_agent.agent_behavior.thresholds
- status: active
- type: guideline
<!-- content -->
Use these thresholds when deciding which recommendations to load:

| Threshold | Action |
|-----------|--------|
| ≥ 0.9     | **Must load** — Critical context |
| 0.7–0.9   | **Should load** — Important context |
| 0.5–0.7   | **Consider loading** — Potentially useful |
| < 0.5     | **Skip** — Low relevance |

### Token Budget Management
- id: meta_mcp_agent.agent_behavior.token_budget
- status: active
- type: guideline
<!-- content -->
Respect context window limits by managing loaded knowledge.

**Strategy:**
1. Sum `estimated_tokens` for all recommendations above threshold
2. If total exceeds budget (e.g., 8000 tokens), prioritize by `relevance_score`
3. For large documents, request specific sections via node ID rather than full file
4. Use `include_children: false` unless child details are necessary

### Citing Knowledge
- id: meta_mcp_agent.agent_behavior.citing
- status: active
- type: guideline
<!-- content -->
When applying knowledge from the knowledge base, cite the source.

**Format:**
```
Per [SECURITY.md#authentication], JWT tokens must include...
```

This enables humans to verify the agent's reasoning and trace decisions back to authoritative sources.

### Feedback Loop
- id: meta_mcp_agent.agent_behavior.feedback
- status: active
- type: guideline
<!-- content -->
If `discover_context` returns no relevant results for a legitimate task, the agent SHOULD:

1. Attempt `search_knowledge` with alternative phrasings
2. If still no results, note the gap: "No knowledge base entry found for [topic]"
3. Proceed using general knowledge but flag uncertainty
4. Optionally suggest creating a new knowledge base entry

## Directory Structure
- id: meta_mcp_agent.directory_structure
- status: active
- type: documentation
<!-- content -->
Recommended project layout for the MCP server implementation.

```
meta_mcp_agent/
├── src/
│   ├── __init__.py
│   ├── server.py           # MCP server entry point
│   ├── index_builder.py    # Knowledge base indexer
│   ├── semantic_matcher.py # Discovery algorithm
│   ├── index_sync.py       # File watching & sync
│   └── md_adapter.py       # Adapter for language/md_parser.py
├── tests/
│   ├── test_discovery.py
│   ├── test_retrieval.py
│   └── fixtures/
├── config/
│   ├── default.yaml        # Default configuration
│   └── boost_matrix.yaml   # Task-type boost weights
├── pyproject.toml
└── README.md
```

## Configuration Schema
- id: meta_mcp_agent.configuration
- status: active
- type: documentation
<!-- content -->
Server configuration options.

```yaml

# config/default.yaml
- type: agent_skill
<!-- content -->
knowledge_base:
  root_path: "./knowledge_bases"
  file_patterns:
    - "**/*.md"
  exclude_patterns:
    - "**/node_modules/**"
    - "**/.git/**"

index:
  storage_path: "./.meta_mcp_index"
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  embedding_dimensions: 384
  similarity_metric: "cosine"

discovery:
  default_max_results: 5
  min_relevance_threshold: 0.3
  boost_active_status: 1.2
  boost_recent_update_days: 30
  boost_recent_update_factor: 1.1

server:
  transport: "stdio"  # or "http"
  host: "127.0.0.1"
  port: 8765
  log_level: "info"
```

## Testing Strategy
- id: meta_mcp_agent.testing
- status: active
- type: plan
<!-- content -->
Validation approach for the implementation.

### Unit Tests
- id: meta_mcp_agent.testing.unit
- status: todo
- type: task
<!-- content -->
Test individual components in isolation.

**Coverage:**
- `index_builder.py`: Parsing, embedding generation, index storage
- `semantic_matcher.py`: Similarity calculation, boost application, deduplication
- `server.py`: Tool input validation, response formatting

### Integration Tests
- id: meta_mcp_agent.testing.integration
- status: todo
- type: task
- blocked_by: [meta_mcp_agent.testing.unit]
<!-- content -->
Test the full discovery → retrieval flow.

**Scenarios:**
1. Task with clear matches → Returns relevant guidelines
2. Task with no matches → Returns empty with appropriate message
3. Task with ambiguous matches → Returns diverse recommendations
4. Large knowledge base → Completes within latency budget (<500ms)

### Golden Tests
- id: meta_mcp_agent.testing.golden
- status: todo
- type: task
- blocked_by: [meta_mcp_agent.testing.integration]
<!-- content -->
Snapshot tests with known-good outputs.

**Setup:**
1. Create a fixture knowledge base with representative content
2. Define a set of task descriptions covering all `task_type` values
3. Record expected recommendations for each
4. Assert output matches golden files (with tolerance for score variance)

## Security Considerations
- id: meta_mcp_agent.security
- status: active
- type: guideline
<!-- content -->
Security boundaries for the MCP server.

### Path Traversal Prevention
- id: meta_mcp_agent.security.path_traversal
- status: active
- type: guideline
<!-- content -->
All file access MUST be restricted to the configured `knowledge_base.root_path`.

**Implementation:**
```python
def validate_path(requested_path: str, root: Path) -> Path:
    resolved = (root / requested_path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise SecurityError("Path traversal attempt detected")
    return resolved
```

### Resource Limits
- id: meta_mcp_agent.security.limits
- status: active
- type: guideline
<!-- content -->
Prevent denial-of-service through resource exhaustion.

**Limits:**
- Max file size to index: 1MB
- Max nodes per file: 500
- Max concurrent requests: 10
- Request timeout: 30s
- Max `ids` per `retrieve_knowledge` call: 20

## Versioning & Compatibility
- id: meta_mcp_agent.versioning
- status: active
- type: documentation
<!-- content -->
Protocol version: `1.0.0`

**Compatibility guarantees:**
- Tool schemas are additive (new optional fields allowed)
- Breaking changes require major version bump
- Deprecations announced one minor version before removal

**MCP SDK compatibility:**
- Minimum: `mcp>=1.0.0`
- Tested with: `mcp==1.2.0`

## Future Extensions
- id: meta_mcp_agent.future
- status: draft
- type: plan
<!-- content -->
Potential enhancements for future versions.

### Multi-Repository Support
- id: meta_mcp_agent.future.multi_repo
- status: draft
- type: task
<!-- content -->
Allow indexing knowledge bases from multiple repositories with namespace isolation.

### Write-Back Capability
- id: meta_mcp_agent.future.write_back
- status: draft
- type: task
<!-- content -->
Enable agents to propose updates to knowledge bases when they discover gaps or outdated information.

### Usage Analytics
- id: meta_mcp_agent.future.analytics
- status: draft
- type: task
<!-- content -->
Track which knowledge bases are most frequently accessed to inform maintenance priorities.

### Embedding Cache Sharing
- id: meta_mcp_agent.future.embedding_cache
- status: draft
- type: task
<!-- content -->
Share embedding caches across team members to reduce cold-start latency.
