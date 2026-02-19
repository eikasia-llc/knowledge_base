# MCP-RL Protocol Optimizer — Implementation Plan
- status: in-progress
- type: plan
- owner: ignacio
- priority: high
- id: mcprl
- last_checked: 2026-02-01
- context_dependencies: {"conventions": "MD_CONVENTIONS.md", "prototype": "mcp_rl_protocol.py"}
- label: ['planning']
<!-- content -->
This plan specifies a **production-grade implementation** of the MCP-RL Protocol Optimizer: a system where an LLM acts as a frozen policy for an MDP, and a Reinforcement Learning loop optimizes the **protocol layer** (the structured context the LLM receives via MCP) to maximize cumulative reward.

The core insight is simple: we do NOT fine-tune the LLM. Instead, we treat the protocol — system prompt, state representation templates, tool definitions, history windowing — as a learnable parameter vector θ, and optimize it using reward signals from an MDP environment.

```
┌─────────────────────────────────────────────────────────────┐
│                      MDP Environment                        │
│   state ──► Protocol(θ) ──► LLM (frozen) ──► action        │
│     ▲                                           │           │
│     └───────────── step(action) ◄───────────────┘           │
│                        │                                    │
│                      reward ──► update θ                    │
└─────────────────────────────────────────────────────────────┘
```

The working prototype in `mcp_rl_protocol.py` demonstrates this idea with a simulated LLM and a GridWorld MDP. This plan specifies how to refactor that prototype into a modular, production-ready codebase that supports real LLM backends, arbitrary MDP environments, and pluggable optimization strategies.

## Theoretical Foundation
- status: active
- type: documentation
- id: mcprl.theory
- last_checked: 2026-02-01
- label: ['planning']
<!-- content -->
This section lays the formal groundwork. A coding agent should read this to understand **why** the architecture works before building it.

### The Composed Policy
- status: active
- type: documentation
- id: mcprl.theory.composed_policy
- last_checked: 2026-02-01
- label: ['planning']
<!-- content -->
Standard RL optimizes a policy π(a|s) directly. Here, our policy is a **composition**:

```
π_θ(a|s) = LLM( Protocol(s; θ) )
```

The LLM is a fixed, non-differentiable function. The protocol `Protocol(s; θ)` is a parameterized context-rendering function that transforms raw MDP state into structured text (system prompt + user message + tool schemas). The parameter vector θ encodes discrete choices (which template variant to use) and continuous weights (inclusion probabilities, emphasis parameters).

Since the LLM is a black box, we cannot backpropagate through it. This forces us into **derivative-free optimization** — evolutionary strategies, bandits, Bayesian optimization, or REINFORCE-style policy gradient with score-function estimators over the discrete protocol space.

### Why This Works: The Protocol Hypothesis
- status: active
- type: documentation
- id: mcprl.theory.protocol_hypothesis
- last_checked: 2026-02-01
- label: ['planning']
<!-- content -->
LLMs are highly sensitive to prompt structure. The same factual content presented differently yields dramatically different reasoning quality. This means the mapping from θ to expected reward has meaningful gradient-like structure in practice — some protocol configurations genuinely produce better actions than others.

Formally, let `R(θ) = E_τ[Σ_t r_t | π_θ]` be the expected return under the composed policy. The protocol hypothesis states that `R(θ)` has sufficient structure (smoothness in the continuous dimensions, clear optima in the discrete dimensions) to be optimizable by derivative-free methods within a tractable number of episodes.

The prototype validates this: evolutionary and bandit optimizers both find protocol configurations that significantly outperform random baselines on the GridWorld MDP.

### Connection to Signaling Games
- status: active
- type: documentation
- id: mcprl.theory.signaling_games
- last_checked: 2026-02-01
- label: ['planning']
<!-- content -->
This framework can be viewed through the lens of **signaling games** from game theory. The protocol acts as a sender that encodes the environment state into a signal (structured text). The LLM acts as a receiver that maps signals to actions. Optimizing θ is equivalent to evolving the sender's encoding strategy to maximize a shared payoff (MDP reward).

This connects to the literature on emergent communication in multi-agent RL, where agents learn signaling protocols through reward. The difference here is that one "agent" (the LLM) is frozen, so the burden of adaptation falls entirely on the protocol layer.

## Architecture Overview
- status: active
- type: documentation
- id: mcprl.architecture
- last_checked: 2026-02-01
- label: ['planning']
<!-- content -->
The system is organized as five independent modules that communicate through well-defined interfaces. Each module is a Python package with its own responsibility.

```
mcprl/
├── environments/        # MDP environment interface + implementations
│   ├── __init__.py
│   ├── base.py          # Abstract MDPEnvironment class
│   ├── gridworld.py     # GridWorld reference implementation
│   └── gym_wrapper.py   # Adapter for Gymnasium environments
├── protocol/            # The learnable protocol layer
│   ├── __init__.py
│   ├── components.py    # ProtocolComponent, MCPToolSchema dataclasses
│   ├── evolvable.py     # EvolvableProtocol class (genome encode/decode)
│   └── renderers.py     # State/history rendering strategies
├── llm/                 # LLM backend interface + implementations
│   ├── __init__.py
│   ├── base.py          # Abstract LLMBackend class
│   ├── simulated.py     # SimulatedLLM for fast testing
│   └── anthropic.py     # Real Claude API backend via MCP
├── optimizers/          # Protocol optimization algorithms
│   ├── __init__.py
│   ├── base.py          # Abstract ProtocolOptimizer class
│   ├── evolutionary.py  # Genetic algorithm optimizer
│   ├── bandit.py        # Component-wise UCB bandit optimizer
│   └── bayesian.py      # Bayesian optimization (stretch goal)
├── training/            # Training loop, logging, analysis
│   ├── __init__.py
│   ├── loop.py          # run_episode, train_protocol
│   ├── logging.py       # Structured experiment logging
│   └── analysis.py      # Post-training analysis and visualization
├── config/              # Configuration and experiment definitions
│   ├── __init__.py
│   └── experiment.py    # ExperimentConfig dataclass
└── main.py              # CLI entry point
```

## Task 1: Environment Module
- status: todo
- type: task
- owner: claude
- estimate: 2h
- id: mcprl.env
- priority: high
- blocked_by: []
- last_checked: 2026-02-01
- label: ['planning']
<!-- content -->
Extract the MDP environment interface and implementations from the prototype into `mcprl/environments/`.

### 1a. Abstract Base Class
- status: todo
- type: task
- owner: claude
- estimate: 30m
- id: mcprl.env.base
- blocked_by: []
- last_checked: 2026-02-01
- label: ['planning']
<!-- content -->
Create `environments/base.py` containing the `MDPEnvironment` abstract base class. Carry over the interface from the prototype (`reset`, `step`, `get_available_actions`) and add two new methods:

```python
@abstractmethod
def get_state_schema(self) -> dict:
    """
    Return a JSON Schema describing the state dict structure.
    
    This enables the protocol layer to auto-generate rendering
    templates for arbitrary environments without hardcoding.
    """
    ...

@abstractmethod
def get_reward_range(self) -> tuple[float, float]:
    """
    Return (min_possible_reward, max_possible_reward) for normalization.
    
    The optimizer uses this to normalize reward signals across
    different environments.
    """
    ...
```

The `step` method signature stays as `step(action: str) -> tuple[dict, float, bool, dict]` to maintain compatibility with MCP tool calling, where actions are always string-typed.

### 1b. GridWorld Reference Implementation
- status: todo
- type: task
- owner: claude
- estimate: 30m
- id: mcprl.env.gridworld
- blocked_by: [mcprl.env.base]
- last_checked: 2026-02-01
- label: ['planning']
<!-- content -->
Move `GridWorldMDP` from the prototype into `environments/gridworld.py`. Implement the two new abstract methods. Add configurable parameters for grid size, trap positions (accept a list), and reward values so the environment can be made harder for more interesting optimization.

### 1c. Gymnasium Wrapper
- status: todo
- type: task
- owner: claude
- estimate: 1h
- id: mcprl.env.gym_wrapper
- blocked_by: [mcprl.env.base]
- last_checked: 2026-02-01
- label: ['planning']
<!-- content -->
Create `environments/gym_wrapper.py` with a `GymMDPWrapper` class that adapts any Gymnasium environment to the `MDPEnvironment` interface.

Key design decisions:

The wrapper must convert Gymnasium's numeric action spaces into string actions for MCP compatibility. For `Discrete(n)` spaces, map indices to descriptive string names (accept a `dict[int, str]` mapping in the constructor, e.g. `{0: "left", 1: "right"}`). For `Box` spaces, this wrapper should raise `NotImplementedError` with a clear message — continuous action spaces require a different approach (discretization or a dedicated protocol component) that is out of scope for the initial release.

State conversion: Gymnasium observations (numpy arrays, dicts, tuples) need to be serialized to `dict` form. For `Dict` spaces, pass through directly. For `Box`/array spaces, convert to `{"observation": list_of_values}`. For `Tuple` spaces, convert to `{"obs_0": ..., "obs_1": ..., ...}`.

```python

# Usage example the implementer should target:
- type: plan
- label: ['planning']
<!-- content -->
import gymnasium as gym

env = GymMDPWrapper(
    gym_env=gym.make("CartPole-v1"),
    action_names={0: "push_left", 1: "push_right"},
)
state = env.reset()           # -> {"observation": [x, x_dot, theta, theta_dot]}
actions = env.get_available_actions()  # -> ["push_left", "push_right"]
```

## Task 2: Protocol Module
- status: todo
- type: task
- owner: claude
- estimate: 3h
- id: mcprl.protocol
- priority: high
- blocked_by: [mcprl.env]
- last_checked: 2026-02-01
- label: ['planning']
<!-- content -->
Extract and extend the protocol layer from the prototype into `mcprl/protocol/`.

### 2a. Component Dataclasses
- status: todo
- type: task
- owner: claude
- estimate: 30m
- id: mcprl.protocol.components
- blocked_by: []
- last_checked: 2026-02-01
- label: ['planning']
<!-- content -->
Create `protocol/components.py` containing `ProtocolComponent` and `MCPToolSchema` dataclasses. These are carried over from the prototype with one addition: each `ProtocolComponent` should include an optional `component_type` enum field to classify it:

```python
class ComponentType(str, Enum):
    """Classification of protocol component roles."""
    INSTRUCTION = "instruction"    # System prompt / persona
    STATE_FORMAT = "state_format"  # How state is rendered
    HISTORY = "history"            # Trajectory windowing
    REWARD_FRAME = "reward_frame"  # How rewards are communicated
    TOOL_SCHEMA = "tool_schema"    # MCP tool definition
    CUSTOM = "custom"              # User-defined extension
```

This classification enables the optimizer to apply type-specific mutation strategies (e.g., always include INSTRUCTION components, allow HISTORY components to be dropped entirely).

### 2b. Renderer Strategies
- status: todo
- type: task
- owner: claude
- estimate: 1h
- id: mcprl.protocol.renderers
- blocked_by: [mcprl.protocol.components, mcprl.env.base]
- last_checked: 2026-02-01
- label: ['planning']
<!-- content -->
Create `protocol/renderers.py` with a `StateRenderer` abstract base class and concrete implementations for each rendering strategy (JSON, markdown table, natural language, ASCII grid).

The critical design point: renderers must be **generic**. The prototype hardcodes GridWorld-specific field names (`x`, `y`, `goal_x`, etc.). The production version should use the environment's `get_state_schema()` to auto-generate rendering logic for arbitrary state dicts.

```python
class StateRenderer(ABC):
    """Converts an MDP state dict into a text representation for the LLM."""

    @abstractmethod
    def render(self, state: dict, schema: dict | None = None) -> str:
        """
        Render a state dict as formatted text.

        Args:
            state:  The raw state dict from the environment.
            schema: Optional JSON Schema from env.get_state_schema().
                    Renderers may use this for auto-formatting.
        """
        ...
```

Implement four concrete renderers matching the prototype: `JsonRenderer`, `MarkdownTableRenderer`, `NaturalLanguageRenderer`, `AsciiGridRenderer`. The `AsciiGridRenderer` is inherently domain-specific (needs x/y coordinates), so it should detect whether the state has positional fields and fall back to `MarkdownTableRenderer` if not.

Additionally, create a `HistoryRenderer` class with variants for "none", "cumulative_only", "per_step", and "trend" framing. This mirrors the prototype's `render_history` method but as a pluggable strategy.

### 2c. EvolvableProtocol
- status: todo
- type: task
- owner: claude
- estimate: 1h
- id: mcprl.protocol.evolvable
- blocked_by: [mcprl.protocol.components, mcprl.protocol.renderers]
- last_checked: 2026-02-01
- label: ['planning']
<!-- content -->
Create `protocol/evolvable.py` with the main `EvolvableProtocol` class. This is the θ in the optimization — it holds all components and knows how to encode/decode itself as a genome dict.

Carry over `get_genome`, `set_genome`, `get_genome_hash`, and `render_full_context` from the prototype. Refactor `render_full_context` to delegate to the renderer strategies from Task 2b instead of using internal methods.

Add a factory method for quick setup:

```python
@classmethod
def from_config(cls, config: dict) -> "EvolvableProtocol":
    """
    Build a protocol from a configuration dict.

    This allows experiment configs to define protocol search spaces
    declaratively, without writing Python code.

    Example config:
    {
        "system_instruction": {
            "variants": ["You are an agent...", "You are a strategic agent..."],
            "inclusion_weight": 1.0
        },
        "state_format": {
            "variants": ["json_raw", "markdown_table", "natural_language"],
            "inclusion_weight": 1.0
        },
        ...
    }
    """
    ...
```

The `render_full_context` method must output a dict matching the Anthropic Messages API structure: `{"system": str, "user": str, "tools": list[dict]}`. This is the bridge between protocol and LLM — the output is directly usable as API call parameters.

## Task 3: LLM Backend Module
- status: todo
- type: task
- owner: claude
- estimate: 3h
- id: mcprl.llm
- priority: high
- blocked_by: [mcprl.protocol]
- last_checked: 2026-02-01
- label: ['planning']
<!-- content -->
Extract the LLM interface and create real + simulated backends in `mcprl/llm/`.

### 3a. Abstract LLM Interface
- status: todo
- type: task
- owner: claude
- estimate: 30m
- id: mcprl.llm.base
- blocked_by: []
- last_checked: 2026-02-01
- label: ['planning']
<!-- content -->
Create `llm/base.py` with an abstract `LLMBackend` class:

```python
class LLMBackend(ABC):
    """
    Abstract interface for an LLM that acts as the frozen policy.

    The LLM receives a structured context (system prompt, user message,
    tool definitions) and returns an action by "calling" one of the
    provided MCP tools.
    """

    @abstractmethod
    def choose_action(
        self,
        context: dict,
        available_actions: list[str],
    ) -> str:
        """
        Choose an action given the rendered protocol context.

        Args:
            context: Dict with keys "system", "user", "tools" — the
                     output of EvolvableProtocol.render_full_context().
            available_actions: List of valid action strings. The
                     implementation should validate that the returned
                     action is in this list.

        Returns:
            A string action name from available_actions.
        """
        ...
```

Note the interface change from the prototype: the `state` parameter is removed. The LLM should only see what the protocol gives it — passing raw state would defeat the purpose of learning the protocol. The simulated LLM is an exception (it needs raw state for its heuristic), handled via its constructor.

### 3b. Simulated LLM
- status: todo
- type: task
- owner: claude
- estimate: 30m
- id: mcprl.llm.simulated
- blocked_by: [mcprl.llm.base]
- last_checked: 2026-02-01
- label: ['planning']
<!-- content -->
Move `SimulatedLLM` from the prototype into `llm/simulated.py`. This stays mostly the same — it accepts raw state in its constructor (via an environment reference or a state callback) and uses context quality heuristics to modulate action noise.

The simulated LLM is essential for development and testing. It must remain fast (no API calls) so that the optimization loop can run thousands of episodes cheaply.

### 3c. Anthropic API Backend
- status: todo
- type: task
- owner: claude
- estimate: 2h
- id: mcprl.llm.anthropic
- priority: high
- blocked_by: [mcprl.llm.base]
- last_checked: 2026-02-01
- label: ['planning']
<!-- content -->
Create `llm/anthropic.py` with an `AnthropicLLMBackend` that calls the Claude API. This is the production backend that makes the system real.

Implementation requirements:

**API call structure.** The `choose_action` method must construct a Messages API call using the context dict. The `context["system"]` becomes the system parameter. The `context["user"]` becomes the user message content. The `context["tools"]` becomes the tools parameter. The model should be configurable (default to `claude-sonnet-4-20250514` for cost-efficiency during optimization).

**Action extraction.** Parse the API response to extract the tool call. The LLM should call the `take_action` tool with a `direction` (or domain-appropriate) parameter. If the response contains no tool call or an invalid action, fall back to a random action from `available_actions` and log a warning.

**Rate limiting and retries.** The training loop may fire hundreds of API calls. Implement exponential backoff with jitter. Accept a `requests_per_minute` parameter in the constructor and enforce it with a token-bucket rate limiter.

**Caching.** Identical (context, state) pairs should return cached responses. Use the protocol's `get_genome_hash()` plus a hash of the user message as the cache key. Store in a local SQLite database so cache persists across runs. This is critical for cost control — many optimization episodes will produce identical contexts.

**Cost tracking.** Log input/output token counts per call. Expose a `get_total_cost()` method that estimates USD spent (using current API pricing). Print a warning if cost exceeds a configurable threshold.

```python
class AnthropicLLMBackend(LLMBackend):
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,     # Reads ANTHROPIC_API_KEY env var if None
        requests_per_minute: int = 40,
        cache_db_path: str = ".mcprl_cache.db",
        cost_warning_usd: float = 5.0,
    ):
        ...
```

## Task 4: Optimizer Module
- status: todo
- type: task
- owner: claude
- estimate: 3h
- id: mcprl.optimizers
- priority: high
- blocked_by: [mcprl.protocol]
- last_checked: 2026-02-01
- label: ['planning']
<!-- content -->
Extract and extend the optimization algorithms into `mcprl/optimizers/`.

### 4a. Optimizer Base Class
- status: todo
- type: task
- owner: claude
- estimate: 20m
- id: mcprl.optimizers.base
- blocked_by: []
- last_checked: 2026-02-01
- label: ['planning']
<!-- content -->
Create `optimizers/base.py` with the `ProtocolOptimizer` abstract class. Carry over `suggest_genome` and `report_reward` from the prototype. Add a `get_best_genome` method:

```python
class ProtocolOptimizer(ABC):

    @abstractmethod
    def suggest_genome(self) -> dict:
        """Propose a protocol genome to evaluate next."""
        ...

    @abstractmethod
    def report_reward(self, genome: dict, total_reward: float):
        """Report the average episode reward for a genome."""
        ...

    @abstractmethod
    def get_best_genome(self) -> dict | None:
        """Return the best genome found so far, or None if no evaluations yet."""
        ...
```

### 4b. Evolutionary Optimizer
- status: todo
- type: task
- owner: claude
- estimate: 1h
- id: mcprl.optimizers.evolutionary
- blocked_by: [mcprl.optimizers.base]
- last_checked: 2026-02-01
- label: ['planning']
<!-- content -->
Move `EvolutionaryOptimizer` into `optimizers/evolutionary.py`. Carry over the full implementation (population management, elitism, mutation) from the prototype.

Add one enhancement: **crossover**. The prototype only uses mutation. Implement single-point crossover where two parent genomes are split at a random gene boundary and recombined. This is standard in genetic algorithms and should improve search efficiency on larger protocol spaces.

```python
def _crossover(self, parent_a: dict, parent_b: dict) -> dict:
    """
    Single-point crossover of two parent genomes.

    Randomly select a split point among the genome keys,
    then take keys before the split from parent_a and
    keys after the split from parent_b.
    """
    ...
```

### 4c. Component Bandit Optimizer
- status: todo
- type: task
- owner: claude
- estimate: 1h
- id: mcprl.optimizers.bandit
- blocked_by: [mcprl.optimizers.base]
- last_checked: 2026-02-01
- label: ['planning']
<!-- content -->
Move `ComponentBanditOptimizer` into `optimizers/bandit.py`. Carry over the UCB1 selection and per-component statistics from the prototype.

Add one enhancement: **contextual bandits**. The current implementation treats each component independently (standard MAB). A contextual extension would condition variant selection on the current environment state summary (e.g., "early game" vs "near goal" vs "near trap"). This is a stretch goal — implement the standard UCB1 first, then add a `ContextualComponentBanditOptimizer` subclass if time allows.

### 4d. Bayesian Optimizer (Stretch Goal)
- status: todo
- type: task
- owner: claude
- estimate: 2h
- id: mcprl.optimizers.bayesian
- priority: low
- blocked_by: [mcprl.optimizers.base]
- last_checked: 2026-02-01
- label: ['planning']
<!-- content -->
Implement a `BayesianOptimizer` in `optimizers/bayesian.py` using a Gaussian Process surrogate model over the genome space. This is a stretch goal because it requires handling the mixed discrete/continuous nature of the genome space, which is non-trivial.

Use `scikit-optimize` (skopt) as the backend if available, with a fallback to a simple random-search baseline if the dependency is not installed.

This optimizer is theoretically the most sample-efficient, which matters when using the real Anthropic API backend (each evaluation costs money).

## Task 5: Training Loop and Logging
- status: todo
- type: task
- owner: claude
- estimate: 2h
- id: mcprl.training
- priority: high
- blocked_by: [mcprl.env, mcprl.protocol, mcprl.llm, mcprl.optimizers]
- last_checked: 2026-02-01
- label: ['planning']
<!-- content -->
Build the training orchestration layer in `mcprl/training/`.

### 5a. Episode Runner and Training Loop
- status: todo
- type: task
- owner: claude
- estimate: 1h
- id: mcprl.training.loop
- blocked_by: [mcprl.env.base, mcprl.protocol.evolvable, mcprl.llm.base, mcprl.optimizers.base]
- last_checked: 2026-02-01
- label: ['planning']
<!-- content -->
Create `training/loop.py` with `run_episode` and `train_protocol` functions. These are refactored versions of the prototype functions with the following changes:

`run_episode` should accept the abstract `LLMBackend` instead of `SimulatedLLM`. Remove the raw `state` parameter from the LLM call — the LLM only sees the rendered context (except for SimulatedLLM which handles this internally).

`train_protocol` should accept an `ExperimentConfig` dataclass (see Task 6) instead of individual parameters. Add support for early stopping: if average reward has not improved for `patience` generations, stop and return.

Add a checkpoint mechanism: every N generations, serialize the optimizer state and best genome to a JSON file so training can be resumed.

### 5b. Structured Logging
- status: todo
- type: task
- owner: claude
- estimate: 30m
- id: mcprl.training.logging
- blocked_by: [mcprl.training.loop]
- last_checked: 2026-02-01
- label: ['planning']
<!-- content -->
Create `training/logging.py` with an `ExperimentLogger` class that records per-generation metrics to both console and a JSON Lines file.

Each log line should include: generation number, average reward, goal rate, trap rate (or domain-appropriate metrics), genome hash, wall-clock time, and (if using Anthropic backend) API cost so far.

The log file should be loadable by the analysis module for post-hoc visualization.

### 5c. Analysis and Visualization
- status: todo
- type: task
- owner: claude
- estimate: 30m
- id: mcprl.training.analysis
- blocked_by: [mcprl.training.logging]
- last_checked: 2026-02-01
- label: ['planning']
<!-- content -->
Create `training/analysis.py` with functions to produce post-training reports. Refactor `analyze_results` from the prototype. Add matplotlib-based plotting (reward curve, goal rate over generations, component selection heatmap for the bandit optimizer).

Output a summary markdown file in the MD_CONVENTIONS format containing the best genome, reward trajectory statistics, and per-component analysis.

## Task 6: Configuration and CLI
- status: todo
- type: task
- owner: claude
- estimate: 1h
- id: mcprl.config
- priority: medium
- blocked_by: [mcprl.training]
- last_checked: 2026-02-01
- label: ['planning']
<!-- content -->

### 6a. Experiment Configuration
- status: todo
- type: task
- owner: claude
- estimate: 30m
- id: mcprl.config.experiment
- blocked_by: []
- last_checked: 2026-02-01
- label: ['planning']
<!-- content -->
Create `config/experiment.py` with an `ExperimentConfig` dataclass:

```python
@dataclass
class ExperimentConfig:
    """Complete specification of an MCP-RL optimization experiment."""

    # -- Environment --
    env_type: str = "gridworld"          # "gridworld" | "gym"
    env_kwargs: dict = field(default_factory=dict)

    # -- LLM Backend --
    llm_backend: str = "simulated"       # "simulated" | "anthropic"
    llm_kwargs: dict = field(default_factory=dict)

    # -- Protocol --
    protocol_config: dict | None = None  # None = use defaults

    # -- Optimizer --
    optimizer_type: str = "evolutionary"  # "evolutionary" | "bandit" | "bayesian"
    optimizer_kwargs: dict = field(default_factory=dict)

    # -- Training --
    n_generations: int = 50
    episodes_per_genome: int = 5
    patience: int = 15                   # Early stopping patience
    checkpoint_every: int = 10
    seed: int = 42

    # -- Output --
    output_dir: str = "./results"
    experiment_name: str = "default"
```

Support loading from YAML files so experiments are reproducible and shareable.

### 6b. CLI Entry Point
- status: todo
- type: task
- owner: claude
- estimate: 30m
- id: mcprl.config.cli
- blocked_by: [mcprl.config.experiment, mcprl.training.loop]
- last_checked: 2026-02-01
- label: ['planning']
<!-- content -->
Create `main.py` with an `argparse`-based CLI:

```bash

# Run with defaults (GridWorld + SimulatedLLM + Evolutionary)
- type: plan
- label: ['planning']
<!-- content -->
python -m mcprl

# Run with a config file
- type: plan
- label: ['planning']
<!-- content -->
python -m mcprl --config experiments/gridworld_bandit.yaml

# Run with the real Anthropic backend
- type: plan
- label: ['planning']
<!-- content -->
python -m mcprl --llm anthropic --model claude-sonnet-4-20250514

# Resume from checkpoint
- type: plan
- label: ['planning']
<!-- content -->
python -m mcprl --resume results/default/checkpoint_gen30.json
```

## Task 7: Integration Tests
- status: todo
- type: task
- owner: claude
- estimate: 2h
- id: mcprl.tests
- priority: medium
- blocked_by: [mcprl.training]
- last_checked: 2026-02-01
- label: ['planning']
<!-- content -->
Create a `tests/` directory with pytest-based tests.

### 7a. Unit Tests
- status: todo
- type: task
- owner: claude
- estimate: 1h
- id: mcprl.tests.unit
- blocked_by: [mcprl.env, mcprl.protocol, mcprl.optimizers]
- last_checked: 2026-02-01
- label: ['planning']
<!-- content -->
Test each module in isolation:

For `environments/`: verify that `GridWorldMDP` returns correct reward on goal/trap, respects boundaries, and terminates on max_steps. Verify `GymMDPWrapper` correctly translates CartPole states and actions.

For `protocol/`: verify genome round-tripping (`get_genome` → `set_genome` → `get_genome` produces identical output). Verify all renderers produce non-empty strings. Verify `render_full_context` returns a dict with the three required keys.

For `optimizers/`: verify `EvolutionaryOptimizer` population evolves (fitness improves over generations with a fixed reward function). Verify `ComponentBanditOptimizer` converges to the best variant when one variant always gives higher reward.

### 7b. Integration Test — Full Loop
- status: todo
- type: task
- owner: claude
- estimate: 1h
- id: mcprl.tests.integration
- blocked_by: [mcprl.tests.unit]
- last_checked: 2026-02-01
- label: ['planning']
<!-- content -->
Run a complete `train_protocol` with `GridWorldMDP` + `SimulatedLLM` + `EvolutionaryOptimizer` for 20 generations and assert that average reward in the last 5 generations is higher than the first 5. This validates that the system actually learns.

Run the same test with `ComponentBanditOptimizer`. Both should show improvement.

## Execution Order
- status: active
- type: documentation
- id: mcprl.execution_order
- last_checked: 2026-02-01
- label: ['planning']
<!-- content -->
The dependency graph dictates the following build order. A coding agent should follow this sequence, running tests at each milestone.

**Phase 1 — Foundations (parallel):** Tasks 1a, 2a, 3a, 4a can all be built simultaneously since they only define abstract interfaces.

**Phase 2 — Core implementations:** Tasks 1b, 2b, 2c, 3b, 4b, 4c. Build the GridWorld environment, renderers, evolvable protocol, simulated LLM, and both optimizers. After this phase, the system is functionally equivalent to the prototype.

**Phase 3 — Training loop:** Tasks 5a, 5b, 5c. Wire everything together. At the end of this phase, run the integration test (Task 7b) against the simulated LLM to validate end-to-end correctness.

**Phase 4 — Production features:** Tasks 1c (Gym wrapper), 3c (Anthropic backend), 6a, 6b. These extend the system to real environments and real LLMs.

**Phase 5 — Polish:** Tasks 4d (Bayesian optimizer), 7a (full unit tests). Nice-to-haves that improve quality but aren't blockers.

```
Phase 1:  [1a] [2a] [3a] [4a]        ← Abstract interfaces (parallel)
              │    │    │    │
Phase 2:  [1b] [2b,2c] [3b] [4b,4c]  ← Core implementations
              │    │      │    │
Phase 3:     [5a] ◄──────┴────┘       ← Training loop
              │
             [5b, 5c]                  ← Logging & analysis
              │
Phase 4:  [1c] [3c] [6a, 6b]          ← Production features
              │
Phase 5:  [4d] [7a, 7b]               ← Polish & tests
```

## Technical Notes for the Coding Agent
- status: active
- type: documentation
- id: mcprl.notes
- last_checked: 2026-02-01
- label: ['planning']
<!-- content -->
**Python version:** 3.11+. Use `match` statements where appropriate, `type` union syntax (`X | Y`), and dataclasses with `slots=True` for performance.

**Dependencies (core, install unconditionally):** `anthropic` (for the API backend), `pyyaml` (for config files). **Dependencies (optional, install if present):** `gymnasium` (for the Gym wrapper), `matplotlib` (for plots), `scikit-optimize` (for Bayesian optimizer).

**Type hints:** All public functions must have complete type annotations. Use `Protocol` (from `typing`) for structural subtyping where it simplifies the interface.

**Docstrings:** Every class and public method must have a docstring. Use Google-style format. Include at least one usage example per class.

**Commenting style:** Inline comments should explain *why*, not *what*. The code should be self-documenting for the *what*.

**Error handling:** Never silently swallow exceptions. LLM API errors should be caught, logged, and retried (for transient errors) or raised with a clear message (for configuration errors). Invalid actions from the LLM should trigger a warning log and a random fallback, never a crash.

**The prototype (`mcp_rl_protocol.py`) is the ground truth** for behavior. When in doubt about how a component should work, match the prototype's behavior first, then extend.
