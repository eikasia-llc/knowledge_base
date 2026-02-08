# Software Engines — Explanation
- status: active
- type: context
- id: engine-explanation
- context_dependencies: {"mcp": "MCP_EXPLANATION.md", "conventions": "MD_CONVENTIONS.md"}
- last_checked: 2026-02-07
<!-- content -->
This document explores the concept of a **software engine** in computer science: what it is, what properties define it, and how this concept maps to the **MCP Client Engine** used in this project. Understanding the engine pattern clarifies *why* the MCP architecture is structured the way it is.

## 1. What Is a Software Engine?
- status: active
- type: context
- id: engine-explanation.definition
<!-- content -->
A **software engine** is a core component that runs continuously (or is always ready to run), mediating the flow of information between inputs, processing logic, and outputs. It encapsulates the "how" of a system — the orchestration loop — so that the rest of the system only needs to interact with a stable interface.

The defining metaphor is deliberate: just as a physical engine converts fuel into motion through a continuous cycle (intake → compression → combustion → exhaust), a software engine converts **requests into results** through a continuous processing cycle.

### 1.1 Core Properties
- status: active
- type: context
- id: engine-explanation.definition.properties
<!-- content -->
Not every module or class qualifies as an "engine." The term implies a specific set of architectural properties:

| Property | Description | Example |
|:---------|:------------|:--------|
| **Continuous Availability** | The engine is always running or always ready to accept input. It does not perform a single task and exit. | A game engine runs every frame; a database engine listens for queries indefinitely. |
| **Mediation** | The engine sits *between* components, routing data from producers to consumers. It is the central switchboard. | A browser engine mediates between HTML/CSS/JS and pixel rendering. |
| **Orchestration Loop** | The engine runs an internal loop — receive input, delegate processing, collect results, emit output, repeat. | A search engine loops: crawl → index → rank → serve. |
| **Abstraction Boundary** | The engine hides complexity. Consumers interact with it through a stable interface without knowing the internals. | A SQL engine hides B-trees, buffer pools, and query plans behind `SELECT * FROM ...`. |
| **Pluggability** | The engine's internals can be swapped without changing the external interface. Components are interchangeable. | A rendering engine can switch from software to GPU rendering; the game code above doesn't change. |

### 1.2 Engine vs. Library vs. Framework
- status: active
- type: context
- id: engine-explanation.definition.comparison
<!-- content -->
These terms are often confused. The key distinction lies in **control flow** and **lifecycle**:

| Concept | Who Controls the Loop? | Lifecycle | Analogy |
|:--------|:----------------------|:----------|:--------|
| **Library** | Your code calls the library. You control when and how. | Stateless or short-lived. | A toolbox — you pick up a hammer when you need it. |
| **Framework** | The framework calls your code (inversion of control). | Managed by the framework. | A factory floor — you plug your machine into the assembly line. |
| **Engine** | The engine runs its own loop and mediates between subsystems. | Long-lived, always active. | A car engine — it runs continuously and you steer, but it does the work. |

A library is passive (you call it). A framework is prescriptive (it calls you). An engine is **active and autonomous** — it runs its own loop, and you feed it inputs or register components.

## 2. Canonical Examples of Software Engines
- status: active
- type: context
- id: engine-explanation.examples
<!-- content -->
The engine pattern appears across nearly every domain of computing. Below are well-known examples, grouped by domain, with emphasis on the mediation and loop properties.

### 2.1 Game Engine
- status: active
- type: context
- id: engine-explanation.examples.game
<!-- content -->
**Examples**: Unity, Unreal Engine, Godot.

A game engine is perhaps the most intuitive example. It runs a **main loop** (the "game loop") that executes every frame (typically 30–60+ times per second):

```
while game_is_running:
    process_input()        # Keyboard, mouse, controller
    update_game_state()    # Physics, AI, animations
    render_frame()         # Draw everything to screen
    play_audio()           # Sound effects, music
```

The engine mediates between: player input → game logic → rendering subsystem → audio subsystem. Game developers write scripts and define assets; the engine orchestrates everything.

**Key engine property**: Pluggability — you can swap the renderer (OpenGL → Vulkan) without rewriting game logic.

### 2.2 Browser Engine (Rendering Engine)
- status: active
- type: context
- id: engine-explanation.examples.browser
<!-- content -->
**Examples**: Blink (Chrome), Gecko (Firefox), WebKit (Safari).

The browser engine transforms HTML, CSS, and JavaScript into visual pixels on screen. Its internal loop processes:

```
receive_html_css_js()     # From network or cache
parse_to_dom_tree()       # Build Document Object Model
compute_layout()          # Calculate positions and sizes
paint_pixels()            # Rasterize to screen
handle_events()           # Click, scroll, resize → re-render
```

The engine mediates between: web content → layout calculation → pixel rendering → user interaction. Web developers write HTML/CSS/JS; the engine handles the enormous complexity of turning that into interactive pages.

**Key engine property**: Abstraction boundary — a web developer never thinks about rasterization or GPU texture uploads.

### 2.3 Database Engine
- status: active
- type: context
- id: engine-explanation.examples.database
<!-- content -->
**Examples**: PostgreSQL, SQLite, DuckDB, InnoDB (MySQL).

A database engine listens for queries and manages the full lifecycle of data storage and retrieval:

```
listen_for_query()        # Accept SQL from a client
parse_and_plan()          # Parse SQL → query plan → optimize
execute_plan()            # Read/write data using indexes, joins
manage_transactions()     # ACID guarantees, locking, WAL
return_results()          # Send structured data back to client
```

The engine mediates between: client application → query optimizer → storage layer → transaction manager. Application developers write SQL; the engine decides *how* to execute it efficiently.

**Key engine property**: Continuous availability — the database is always listening, always ready. You never "start" a query manually at the storage level.

### 2.4 Search Engine
- status: active
- type: context
- id: engine-explanation.examples.search
<!-- content -->
**Examples**: Elasticsearch, Apache Solr, Google Search (at a much larger scale).

A search engine maintains an inverted index and responds to queries in real-time:

```
# Indexing loop (background, continuous)
crawl_or_ingest_data()    # Receive new documents
tokenize_and_analyze()    # Break text into searchable terms
update_inverted_index()   # Map terms → document locations

# Query loop (foreground, on-demand)
receive_query()           # User search string
match_and_rank()          # Find relevant documents, score them
return_results()          # Ordered list of matches
```

The engine mediates between: raw documents → index structure → relevance ranking → user results. Notably, it runs **two loops** — one for ingestion and one for querying — both always active.

**Key engine property**: Mediation — the search engine is the intermediary between raw unstructured data and structured, ranked answers.

### 2.5 Rule Engine / Business Rule Engine
- status: active
- type: context
- id: engine-explanation.examples.rules
<!-- content -->
**Examples**: Drools (Java), Easy Rules, CLIPS.

A rule engine evaluates a set of declared rules against incoming data or events:

```
load_rules()              # If-then rules from configuration
receive_facts()           # New data or events
match_rules_to_facts()    # Pattern matching (Rete algorithm)
fire_triggered_rules()    # Execute actions for matched rules
update_working_memory()   # Modify state for next cycle
```

The engine mediates between: business rules (declared by analysts) → incoming events → computed decisions. The rules are declarative ("if order > $1000, apply discount"); the engine handles matching and execution.

**Key engine property**: Pluggability — rules can be added, removed, or changed without modifying the engine code.

### 2.6 Physics Engine
- status: active
- type: context
- id: engine-explanation.examples.physics
<!-- content -->
**Examples**: Box2D, Bullet, PhysX, Havok.

A physics engine simulates physical interactions (collision, gravity, friction) each timestep:

```
for each timestep:
    detect_collisions()       # Broad phase + narrow phase
    resolve_constraints()     # Apply joints, contacts, friction
    integrate_forces()        # Update velocities and positions
    report_events()           # Notify game logic of collisions
```

The engine mediates between: game objects with physical properties → mathematical simulation → updated positions and states. Developers describe objects ("this box has mass 5kg"); the engine computes what happens.

**Key engine property**: Orchestration loop — the engine steps forward in time, continuously, frame by frame.

### 2.7 Workflow / Orchestration Engine
- status: active
- type: context
- id: engine-explanation.examples.workflow
<!-- content -->
**Examples**: Apache Airflow, Temporal, Prefect, AWS Step Functions.

A workflow engine manages the execution of multi-step processes (DAGs of tasks):

```
load_workflow_definition()    # DAG of tasks with dependencies
monitor_task_readiness()      # Are all upstream tasks complete?
dispatch_ready_tasks()        # Send to workers / executors
collect_results()             # Mark complete, handle failures
advance_workflow_state()      # Move to next stage, repeat
```

The engine mediates between: workflow definitions (declared by engineers) → task executors → state tracking → completion. Engineers define *what* should happen and in what order; the engine manages *when* and *how* it executes.

**Key engine property**: Continuous availability + Orchestration loop — Airflow's scheduler runs 24/7, checking for tasks to trigger.

## 3. The Common Pattern
- status: active
- type: context
- id: engine-explanation.pattern
<!-- content -->
Across all examples, the engine pattern follows a common structural template:

```
┌─────────────────────────────────────────────────────────────┐
│                        ENGINE                                │
│                                                              │
│   ┌──────────┐     ┌──────────────┐     ┌──────────────┐   │
│   │  INPUT    │────▶│  PROCESSING  │────▶│   OUTPUT     │   │
│   │ Interface │     │  Loop        │     │  Interface   │   │
│   └──────────┘     └──────┬───────┘     └──────────────┘   │
│                           │                                  │
│                    ┌──────▼───────┐                          │
│                    │  SUBSYSTEMS  │                          │
│                    │  (pluggable) │                          │
│                    └──────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

The abstract engine lifecycle:

1. **Initialize** — Load configuration, register subsystems, prepare state.
2. **Listen / Accept** — Wait for input (query, event, frame tick, request).
3. **Delegate** — Route the input to the appropriate subsystem(s).
4. **Collect** — Gather results from subsystems.
5. **Emit** — Return the output through the output interface.
6. **Repeat** — Go back to step 2. The engine never exits this loop voluntarily.

## 4. The MCP Client Engine as an Instance of This Pattern
- status: active
- type: context
- id: engine-explanation.mcp-engine
<!-- content -->
With the general engine pattern now defined, we can see that the **MCP Client Engine** (`src/core/engine.py`) is a concrete instance of this architecture:

| General Engine Property | MCP Client Engine Implementation |
|:------------------------|:---------------------------------|
| **Continuous Availability** | The engine is instantiated at application startup and remains active for the session, ready to process any user query. |
| **Mediation** | It sits between the **user** (input), the **LLM** (reasoning), and the **MCP Server** (tool execution), routing data between all three. |
| **Orchestration Loop** | The tool-call loop: send query to LLM → receive tool call request → execute tool via MCP Server → feed result back to LLM → repeat until LLM produces final answer. |
| **Abstraction Boundary** | The user sees only a chat interface. The LLM sees only tool schemas. Neither knows about JSON files, SQL queries, or internal routing. |
| **Pluggability** | The LLM provider can be swapped (Gemini ↔ Claude ↔ OpenAI). The data backend can be swapped (JSON ↔ DuckDB). The tool set can be extended. None of these changes affect the engine's loop. |

### 4.1 The MCP Engine Loop in Detail
- status: active
- type: context
- id: engine-explanation.mcp-engine.loop
<!-- content -->
Mapping the abstract engine lifecycle (Section 3) to the MCP Client Engine:

```python
# ─── Abstract Engine Lifecycle → MCP Client Engine ───

class MCPClientEngine:
    """
    Concrete instance of the Software Engine pattern.
    Mediates between User, LLM, and MCP Server.
    """

    def __init__(self, llm_provider, mcp_server):
        # STEP 1: Initialize
        # Register subsystems (LLM + MCP Server with its tools)
        self.llm = llm_provider          # Pluggable: Gemini, Claude, OpenAI
        self.mcp = mcp_server            # Pluggable: JSON-backed, DB-backed

    def run(self, user_query: str) -> str:
        """
        The engine's orchestration loop for a single query.
        In a chat application, this method is called for each user message,
        making the engine effectively 'always listening'.
        """

        # STEP 2: Listen / Accept
        # The user_query arrives from the chat interface.

        # STEP 3: Delegate (first pass — send to LLM with tool schemas)
        tools = self.mcp.list_tools()
        response = self.llm.generate(
            messages=[{"role": "user", "content": user_query}],
            tools=tools
        )

        # STEP 4: Collect — The inner tool-call loop
        # This is the engine's core mediation cycle:
        #   LLM requests tool → Engine executes via MCP → Engine feeds back
        while response.has_tool_calls():
            for call in response.tool_calls:
                # Delegate to MCP Server subsystem
                result = self.mcp.call_tool(call.name, call.arguments)
                # Feed result back to LLM subsystem
                response = self.llm.continue_with_tool_result(result)

        # STEP 5: Emit — Return final answer to the user
        return response.text

        # STEP 6: Repeat — The chat application calls run() again
        # for the next user message. The engine persists across calls.
```

### 4.2 Where the MCP Engine Fits Among Other Engines
- status: active
- type: context
- id: engine-explanation.mcp-engine.positioning
<!-- content -->
| Engine Type | What It Mediates | Loop Trigger | Closest Analog |
|:------------|:-----------------|:-------------|:---------------|
| Game Engine | Input → Logic → Rendering | Frame tick (16ms) | — |
| Database Engine | SQL → Optimizer → Storage | Incoming query | MCP Engine (query-driven) |
| Search Engine | Query → Index → Ranking | Incoming query | MCP Engine (query-driven) |
| Rule Engine | Events → Rules → Actions | Incoming event | MCP Engine (event = user message) |
| Workflow Engine | Task graph → Executors → State | Scheduler tick | MCP Engine (tool calls = tasks) |
| **MCP Client Engine** | **User → LLM → Tools** | **User message** | Database + Workflow hybrid |

The MCP Client Engine is most analogous to a **database engine** (query-driven, mediates between caller and data) crossed with a **workflow engine** (orchestrates a DAG of tool calls where results feed into subsequent decisions).

## 5. Why Call It an "Engine"?
- status: active
- type: context
- id: engine-explanation.why-engine
<!-- content -->
The term is justified because the MCP Client component exhibits all five core properties (Section 1.1). If it were merely a function that called the LLM once, it would be a **library call**. If it imposed a rigid structure that your code had to fit into, it would be a **framework**. But the MCP Client Engine:

1. **Runs its own loop** — the tool-call cycle is internal and autonomous.
2. **Is always available** — instantiated at startup, persists across queries.
3. **Mediates** — it is the central switchboard between three subsystems (user, LLM, tools).
4. **Hides complexity** — the user and the LLM both interact through clean interfaces.
5. **Supports pluggable subsystems** — LLM providers, data backends, and tools are all swappable.

It is not just a helper function. It is the **continuously running core** that makes the system work.

## 6. Summary
- status: active
- type: context
- id: engine-explanation.summary
<!-- content -->
| Concept | Description |
|:--------|:------------|
| **Software Engine** | A continuously available core component that mediates information flow through an orchestration loop, hiding internal complexity behind stable interfaces. |
| **Core Properties** | Continuous availability, mediation, orchestration loop, abstraction boundary, pluggability. |
| **Engine vs. Library** | A library is passive (you call it). An engine is active (it runs its own loop). |
| **Engine vs. Framework** | A framework prescribes structure (inversion of control). An engine orchestrates subsystems autonomously. |
| **Common Examples** | Game engines, browser engines, database engines, search engines, rule engines, workflow engines, physics engines. |
| **MCP Client Engine** | A concrete instance of the engine pattern: mediates between User, LLM, and MCP Tools through a tool-call orchestration loop. |
| **Closest Analogs** | Database engine (query-driven mediation) + workflow engine (DAG of dependent tool calls). |
