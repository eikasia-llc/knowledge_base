# Tunable Layer Guideline: Improving LLM Performance in the Unified Nexus
- status: active
- type: guideline
- id: tunable-layer-guideline
- last_checked: 2026-02-08
- label: [guide]
<!-- content -->
This document describes strategies for adding **trainable intermediate layers** to the Unified Nexus Architecture in order to improve LLM performance for end users. The core idea is that instead of relying solely on prompt engineering or expensive full model retraining, we inject lightweight, fine-tunable components at strategic points in the pipeline. These components can learn from user feedback over time, creating a continuous improvement loop.

Three intervention points are identified, ordered by **effort-to-impact ratio** (best first):

1. **Learned Query Router** — Replace the keyword-based router with a trained classifier.
2. **LoRA Adapters for Tool Calling** — Fine-tune lightweight adapters on MCP tool-use traces.
3. **Preference-Based Optimization** — Use DPO or reward models to align responses with user preferences.

## Literature Overview
- status: active
- type: documentation
- id: tunable-layer-guideline.literature
- last_checked: 2026-02-08
<!-- content -->
The following recent publications directly inform this guideline:

| Paper / Project | Year | Venue | Key Contribution |
|:----------------|:-----|:------|:-----------------|
| **RAGRouter** (Zhang et al.) | 2025 | NeurIPS | Contrastive-learning router that models how retrieved docs shift LLM capabilities. +3.6% over best single LLM. |
| **Router-R1** (Zhang et al.) | 2025 | NeurIPS | RL-based multi-round router that interleaves "think" and "route" actions. |
| **AgentFlux** (Kadekodi et al.) | 2025 | arXiv | Decoupled LoRA fine-tuning for MCP tool calling: separate adapters for tool selection vs. argument generation. +46% accuracy on MCP-Bench. |
| **ScaleMCP** (Lumer et al.) | 2025 | arXiv | Agentic-RAG approach to tool retrieval when tool sets exceed context limits. |
| **MOLoRA** (Hao et al.) | 2024 | arXiv | LoRA adapters specifically designed for multi-tool orchestration. |
| **Online RLHF** (Li et al.) | 2025 | arXiv | One-pass reward modeling with constant-time updates per iteration. |
| **DPO** (Rafailov et al.) | 2023 | NeurIPS | Direct Preference Optimization — learns from preference pairs without a separate reward model. |

## Architectural Placement
- status: active
- type: documentation
- id: tunable-layer-guideline.placement
- last_checked: 2026-02-08
<!-- content -->
The three tunable layers map to specific points in the existing Unified Nexus pipeline. The diagram below shows where each intervention sits relative to the components defined in `KG_UNIFIED_NEXUS_ARCHITECTURE_PLAN.md`.

```
┌─────────────────────────────────────────────────────┐
│                   User Question                      │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│   🎯 TUNABLE LAYER 1: Learned Query Router           │
│   Small classifier (sentence-transformer + head)     │
│   Trained on (query, best_path) logs                 │
│   Replaces keyword-based QueryRouter                 │
└──────────────────────┬──────────────────────────────┘
                       │
      ┌────────────────┼────────────────┐
      ▼                ▼                ▼
  ChromaDB          DuckDB          Graph DB
  (Unstructured)    (Structured)    (Relationships)
      │                │                │
      └────────────────┼────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│   🎯 TUNABLE LAYER 2: LoRA Adapter(s)               │
│   On your local LLM (Qwen, Llama, etc.)             │
│   - Tool selection adapter (which MCP tool to call)  │
│   - Argument generation adapter (correct params)     │
│   - Domain-specific response adapter                 │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│   🎯 TUNABLE LAYER 3: Preference Optimization       │
│   Lightweight feedback loop:                         │
│   - Log (query, response, user_feedback) triples     │
│   - DPO training on preference pairs                 │
│   - Or: reward model scoring for best-of-N reranking │
└─────────────────────────────────────────────────────┘
```

## Layer 1: Learned Query Router
- status: active
- type: guideline
- id: tunable-layer-guideline.router
- priority: high
- last_checked: 2026-02-08
<!-- content -->
The Query Router is the **highest-leverage, lowest-effort** intervention point. It determines which retrieval path (vector search, SQL, graph, hybrid) handles each user question. Currently, this is done with keyword matching or a cheap LLM call. A trained classifier can be significantly more accurate while remaining fast and cheap.

### Approach: Sentence-Transformer Classifier
- status: active
- type: documentation
- id: tunable-layer-guideline.router.approach
- last_checked: 2026-02-08
<!-- content -->
The idea (following practical RAG router guides and the RAGRouter paper) is to fine-tune a small sentence-transformer model with a classification head on top. The model takes a user query as input and outputs a probability distribution over the available retrieval paths.

**Architecture:**

```
User Query (text)
       │
       ▼
┌──────────────────────┐
│  Sentence-Transformer │  ← e.g., all-MiniLM-L6-v2 (22M params)
│  (frozen or fine-tuned)│
└──────────┬───────────┘
           │
           ▼
     [CLS] embedding (384-dim)
           │
           ▼
┌──────────────────────┐
│  Classification Head  │  ← Linear(384, num_classes) + softmax
│  (trained)            │
└──────────┬───────────┘
           │
           ▼
  {UNSTRUCTURED: 0.1, STRUCTURED: 0.05, GRAPH: 0.02, HYBRID: 0.83}
```

**Why this works:** Sentence-transformers already encode semantic meaning. The classification head only needs to learn the mapping from query semantics to retrieval path — a very low-dimensional problem.

### Training Data Collection
- status: active
- type: documentation
- id: tunable-layer-guideline.router.data
- last_checked: 2026-02-08
<!-- content -->
Training data is generated from the system's own usage logs. Each log entry records:

1. The user's original query.
2. Which retrieval path(s) the system used.
3. Whether the final answer was satisfactory (user feedback or heuristic).

The label is the path that produced the best result. Over time, this creates a self-improving dataset.

**Minimum viable dataset:** ~200-500 labeled examples are sufficient for a sentence-transformer classifier (thanks to transfer learning). This can be bootstrapped with synthetic data generated by an LLM:

```python
# Example: generate synthetic training data
# Each entry maps a query to its ideal retrieval path

SYNTHETIC_EXAMPLES = [
    # STRUCTURED path — precise data lookups, aggregations, counts
    ("How many events happened in Q3?", "STRUCTURED"),
    ("What is the average attendance per seminar?", "STRUCTURED"),
    ("List all faculty hired after 2020", "STRUCTURED"),

    # UNSTRUCTURED path — semantic search, policy, context
    ("What is the department's position on open access?", "UNSTRUCTURED"),
    ("Summarize the latest research on modal logic", "UNSTRUCTURED"),
    ("What does the PTO policy say about holidays?", "UNSTRUCTURED"),

    # GRAPH path — relationships, connections, paths
    ("Who collaborates with Prof. Smith?", "GRAPH"),
    ("What is the chain of supervision for PhD students?", "GRAPH"),
    ("Which researchers are connected to the logic group?", "GRAPH"),

    # HYBRID path — requires multiple sources
    ("Which researcher with the most publications also teaches logic?", "HYBRID"),
    ("Compare our event attendance trends with what the report says", "HYBRID"),
    ("Find students who mentioned funding concerns and have > 2 papers", "HYBRID"),
]
```

### Training Script Skeleton
- status: active
- type: documentation
- id: tunable-layer-guideline.router.training
- last_checked: 2026-02-08
<!-- content -->
The following sketch shows how to train a router classifier using the `SetFit` library (few-shot fine-tuning of sentence-transformers). SetFit is recommended because it works well with very small datasets (as few as 8 examples per class).

```python
"""
Router classifier training using SetFit.

SetFit fine-tunes a sentence-transformer with contrastive learning
on a small labeled dataset, then trains a classification head.
Works well with as few as 8 examples per class.

Requirements:
    pip install setfit sentence-transformers

Usage:
    python train_router.py --data router_labels.csv --output models/router
"""

from setfit import SetFitModel, SetFitTrainer
from datasets import Dataset

# -- 1. Load labeled data --------------------------------------------------
# Format: [{"text": "user query", "label": 0}, ...]
# Labels: 0=UNSTRUCTURED, 1=STRUCTURED, 2=GRAPH, 3=HYBRID
train_data = Dataset.from_dict({
    "text": [ex[0] for ex in SYNTHETIC_EXAMPLES],
    "label": [LABEL_MAP[ex[1]] for ex in SYNTHETIC_EXAMPLES],
})

# -- 2. Initialize model ---------------------------------------------------
# Start from a small, fast sentence-transformer
model = SetFitModel.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2",
    labels=["UNSTRUCTURED", "STRUCTURED", "GRAPH", "HYBRID"],
)

# -- 3. Train --------------------------------------------------------------
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_data,
    num_iterations=20,        # Number of contrastive learning pairs
    batch_size=16,
    num_epochs=1,
)
trainer.train()

# -- 4. Save and use -------------------------------------------------------
model.save_pretrained("models/router")

# Inference (< 5ms on CPU)
prediction = model.predict(["How many seminars last month?"])
# → ["STRUCTURED"]
```

### Integration with Unified Engine
- status: active
- type: documentation
- id: tunable-layer-guideline.router.integration
- last_checked: 2026-02-08
<!-- content -->
The trained router replaces the existing keyword-based `classify()` method in the `QueryRouter` class. The integration is a drop-in replacement:

```python
"""
Updated QueryRouter that uses the trained SetFit classifier
instead of keyword matching.
"""

from setfit import SetFitModel
from enum import Enum

class QueryType(Enum):
    UNSTRUCTURED = 0
    STRUCTURED = 1
    GRAPH = 2
    HYBRID = 3

class LearnedQueryRouter:
    """
    Replaces the keyword-based QueryRouter with a trained classifier.

    The model is loaded once at startup and provides sub-5ms inference
    on CPU, making it suitable for real-time routing decisions.
    """

    def __init__(self, model_path: str = "models/router"):
        # Load the trained SetFit model (sentence-transformer + head)
        self.model = SetFitModel.from_pretrained(model_path)

    def classify(self, query: str) -> QueryType:
        """
        Classify a user query into its optimal retrieval path.

        Args:
            query: The user's natural language question.

        Returns:
            QueryType enum indicating which retrieval path to use.
        """
        prediction = self.model.predict([query])[0]
        return QueryType(prediction)
```

## Layer 2: LoRA Adapters for MCP Tool Calling
- status: active
- type: guideline
- id: tunable-layer-guideline.lora
- priority: medium
- last_checked: 2026-02-08
<!-- content -->
When using a **local LLM** as the generation backbone (e.g., Qwen-2.5-7B, Llama-3-8B), LoRA (Low-Rank Adaptation) adapters provide a way to improve tool calling accuracy without retraining the entire model. This is especially relevant for improving how the LLM interacts with MCP tools.

### Core Concept: Decoupled Fine-Tuning (AgentFlux)
- status: active
- type: documentation
- id: tunable-layer-guideline.lora.decoupled
- last_checked: 2026-02-08
<!-- content -->
The AgentFlux paper (2025) demonstrates that tool calling can be disaggregated into two independent subtasks, each fine-tuned with its own LoRA adapter:

1. **Tool Selection Adapter** — Given a user query and the list of available tools, predict which tool to call. This adapter is trained with a loss mask that only penalizes errors in the tool name portion of the output.

2. **Argument Generation Adapter** — Given a user query and the selected tool's schema, generate the correct arguments. This adapter is trained with a loss mask that only penalizes errors in the argument portion.

```
                     User Query + Tool Schemas
                              │
                              ▼
                   ┌─────────────────────┐
                   │   Base LLM Weights   │  ← Frozen (e.g., Qwen-2.5-7B)
                   │   (not modified)     │
                   └─────────┬───────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼                             ▼
   ┌──────────────────┐          ┌──────────────────┐
   │  LoRA Adapter A   │          │  LoRA Adapter B   │
   │  Tool Selection   │          │  Arg Generation   │
   │  (~1-5 MB)        │          │  (~1-5 MB)        │
   └────────┬─────────┘          └────────┬─────────┘
            │                             │
            ▼                             ▼
   "search_people"               {"query": "Logic",
                                  "role_filter": "faculty"}
```

**Why decoupled?** The AgentFlux experiments show that training separate adapters with task-specific loss masking yields 2× the improvement compared to standard fine-tuning on the same dataset. The intuition is that tool selection and argument generation are cognitively different tasks — one is classification, the other is structured generation.

### Training Data Format
- status: active
- type: documentation
- id: tunable-layer-guideline.lora.data
- last_checked: 2026-02-08
<!-- content -->
Training data for LoRA adapters is collected from MCP tool-calling traces. Each trace captures a complete tool interaction:

```json
{
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant with access to the following tools: ..."
        },
        {
            "role": "user",
            "content": "Who works on Logic at the department?"
        },
        {
            "role": "assistant",
            "content": null,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "search_people",
                        "arguments": "{\"query\": \"Logic\", \"role_filter\": \"faculty\"}"
                    }
                }
            ]
        },
        {
            "role": "tool",
            "content": "[{\"name\": \"Prof. Smith\", \"role\": \"faculty\", ...}]"
        },
        {
            "role": "assistant",
            "content": "Based on the search results, Prof. Smith works on Logic..."
        }
    ]
}
```

**Data sources:**

- **Production logs** — Record all MCP interactions from the live system (with user consent).
- **Synthetic generation** — Use a strong model (e.g., Claude, GPT-4) to generate diverse tool-calling scenarios from your tool schemas.
- **Negative examples** — Include cases where the model should NOT call a tool (direct answer from RAG context).

A minimum of ~500-1000 traces is recommended for meaningful LoRA adaptation.

### LoRA Training Sketch
- status: active
- type: documentation
- id: tunable-layer-guideline.lora.training
- last_checked: 2026-02-08
<!-- content -->
The following sketch uses HuggingFace PEFT + TRL for LoRA fine-tuning. QLoRA (4-bit quantized base model) is used to enable training on consumer GPUs (16-24 GB VRAM).

```python
"""
LoRA fine-tuning for MCP tool calling.

Uses QLoRA (4-bit quantization) to fit on a single consumer GPU.
Trains a LoRA adapter on top of a frozen base model.

Requirements:
    pip install transformers peft trl bitsandbytes datasets

Hardware:
    - Minimum: 1x GPU with 16 GB VRAM (e.g., RTX 4080)
    - Recommended: 1x GPU with 24 GB VRAM (e.g., RTX 4090)
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# -- 1. Load base model in 4-bit (QLoRA) -----------------------------------
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",               # Normalized float 4-bit
    bnb_4bit_compute_dtype="float16",         # Compute in fp16
    bnb_4bit_use_double_quant=True,           # Double quantization saves ~0.4 bits/param
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=quantization_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# -- 2. Configure LoRA adapter ---------------------------------------------
lora_config = LoraConfig(
    r=16,                     # Rank — controls adapter capacity (8-64 typical)
    lora_alpha=32,            # Scaling factor (usually 2*r)
    target_modules=[          # Which layers to adapt
        "q_proj", "k_proj",   # Attention query and key projections
        "v_proj", "o_proj",   # Attention value and output projections
    ],
    lora_dropout=0.05,        # Regularization
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
# This adds ~8M trainable params on top of the frozen 7B base

# -- 3. Train ---------------------------------------------------------------
training_config = SFTConfig(
    output_dir="models/tool_calling_adapter",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
)

trainer = SFTTrainer(
    model=model,
    args=training_config,
    train_dataset=tool_calling_dataset,       # The formatted traces from above
    tokenizer=tokenizer,
)
trainer.train()

# -- 4. Save adapter (small file, ~20-50 MB) --------------------------------
model.save_pretrained("models/tool_calling_adapter")
# The base model weights are NOT saved — only the LoRA delta matrices
```

### Applicability Note: Cloud vs. Local LLMs
- status: active
- type: documentation
- id: tunable-layer-guideline.lora.cloud-vs-local
- last_checked: 2026-02-08
<!-- content -->
> [!WARNING]
> LoRA adapters can only be applied to models you control (local deployment). If using a cloud API (Gemini, Claude, OpenAI), you cannot inject LoRA adapters.
>
> **For cloud-based setups**, the equivalent strategy is:
> - **Optimized system prompts** (prompt engineering, few-shot examples).
> - **Best-of-N reranking** — Generate N candidate responses and use a local reward model to pick the best one (see Layer 3).
> - **Provider fine-tuning APIs** — Some providers (OpenAI, Google) offer fine-tuning endpoints, but these are less flexible than LoRA.

## Layer 3: Preference-Based Optimization (DPO / Reward Models)
- status: active
- type: guideline
- id: tunable-layer-guideline.preference
- priority: medium
- last_checked: 2026-02-08
<!-- content -->
This layer closes the feedback loop: it uses signals from real user interactions to continuously improve the system's response quality. Two approaches are relevant, and **DPO is recommended** for this architecture due to its simplicity.

### Option A: Direct Preference Optimization (DPO) — Recommended
- status: active
- type: documentation
- id: tunable-layer-guideline.preference.dpo
- last_checked: 2026-02-08
<!-- content -->
DPO learns directly from preference pairs (chosen vs. rejected responses) without training a separate reward model. This makes it significantly simpler to implement than full RLHF.

**Data collection workflow:**

```
User asks question
        │
        ▼
System generates response
        │
        ▼
User provides feedback (👍 / 👎)
        │
        ▼
┌───────────────────────────────────────┐
│  Preference Logger                     │
│  Stores:                               │
│  - query (str)                         │
│  - chosen_response (str)   ← 👍 rated  │
│  - rejected_response (str) ← 👎 rated  │
│  or: two responses ranked by the user  │
└───────────────────────────────────────┘
        │
        ▼
  Periodic DPO training (e.g., weekly)
```

**DPO training sketch:**

```python
"""
DPO training on collected preference pairs.

DPO directly optimizes the policy model using preference pairs
without needing a separate reward model, which simplifies the
pipeline considerably compared to full RLHF (PPO).

Requirements:
    pip install trl transformers peft

Reference:
    Rafailov et al., "Direct Preference Optimization", NeurIPS 2023
"""

from trl import DPOTrainer, DPOConfig

# Assumes model and tokenizer are already loaded (with LoRA if applicable)

dpo_config = DPOConfig(
    output_dir="models/dpo_adapter",
    beta=0.1,                  # Controls preference strength (0.1-0.5)
    num_train_epochs=1,        # DPO converges quickly
    per_device_train_batch_size=2,
    learning_rate=5e-5,        # Lower LR than SFT
)

trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=preference_dataset,  # Must have: prompt, chosen, rejected
    tokenizer=tokenizer,
)
trainer.train()
```

### Option B: Reward Model for Best-of-N Reranking
- status: active
- type: documentation
- id: tunable-layer-guideline.preference.reranking
- last_checked: 2026-02-08
<!-- content -->
When using a **cloud LLM** that cannot be directly fine-tuned, a local reward model can still improve output quality by scoring multiple candidates and selecting the best one.

**How it works:**

1. The cloud LLM generates N candidate responses (e.g., N=3-5, using temperature > 0).
2. A small local reward model scores each candidate.
3. The highest-scoring candidate is returned to the user.

```
User Query
    │
    ▼
Cloud LLM (temperature=0.7) ──→ [Response A, Response B, Response C]
    │
    ▼
┌────────────────────────────┐
│  Local Reward Model         │
│  (trained on user prefs)    │
│  Scores: A=0.82, B=0.45,   │
│          C=0.91             │
└────────────┬───────────────┘
             │
             ▼
        Response C (best) → User
```

**Reward model architecture:** A sentence-transformer with a regression head (single scalar output), trained on the same preference data as DPO. The difference is that instead of optimizing the generator, we train a scorer that works at inference time.

This approach has higher latency (N× the LLM calls) but works with any cloud API without modification.

### Feedback Data Schema
- status: active
- type: guideline
- id: tunable-layer-guideline.preference.schema
- last_checked: 2026-02-08
- label: [protocol]
<!-- content -->
All user feedback must be logged in a consistent format to enable both DPO training and reward model training. The following schema is recommended:

```json
{
    "id": "feedback-20260208-001",
    "timestamp": "2026-02-08T14:30:00Z",
    "query": "Who collaborates with Prof. Smith on logic?",
    "retrieval_path": "HYBRID",
    "context_used": ["chunk_42", "sql_result_7", "graph_subgraph_3"],
    "response_a": "Prof. Smith collaborates with Dr. Jones and Dr. Brown...",
    "response_b": "Based on our records, several researchers...",
    "preferred": "a",
    "feedback_type": "comparison",
    "user_comment": null
}
```

Fields `response_a`, `response_b`, and `preferred` are the minimum required for DPO training. The additional fields enable richer analysis (e.g., detecting which retrieval paths produce better responses).

## Implementation Roadmap
- status: active
- type: plan
- id: tunable-layer-guideline.roadmap
- last_checked: 2026-02-08
<!-- content -->
The following phased approach is recommended, ordered by effort-to-impact ratio.

### Phase 1: Learned Router
- status: todo
- type: task
- id: tunable-layer-guideline.roadmap.phase1
- priority: high
- estimate: 1d
- blocked_by: []
<!-- content -->
**Goal:** Replace keyword-based `QueryRouter.classify()` with a trained SetFit classifier.

**Steps:**

1. Generate synthetic training data (~200 examples per class) using a strong LLM.
2. Label 50-100 real queries from production logs (manual or LLM-assisted).
3. Train a SetFit model on `all-MiniLM-L6-v2`.
4. Evaluate accuracy on a held-out test set (target: >85%).
5. Integrate into `UnifiedEngine` as a drop-in replacement.

**Cost:** Free (runs on CPU, no GPU needed for training or inference).

### Phase 2: Feedback Logging Infrastructure
- status: todo
- type: task
- id: tunable-layer-guideline.roadmap.phase2
- priority: high
- estimate: 4h
- blocked_by: []
<!-- content -->
**Goal:** Instrument the system to collect preference data from users.

**Steps:**

1. Add 👍/👎 buttons to the Streamlit UI.
2. Log all interactions with the feedback schema defined above.
3. Store logs in a DuckDB table (`feedback_log`) for easy querying.
4. Build a simple dashboard to monitor feedback volume and distribution.

**Cost:** Free (uses existing DuckDB infrastructure).

### Phase 3: LoRA Fine-Tuning (Local LLM only)
- status: todo
- type: task
- id: tunable-layer-guideline.roadmap.phase3
- priority: medium
- estimate: 2d
- blocked_by: [tunable-layer-guideline.roadmap.phase2]
<!-- content -->
**Goal:** Train LoRA adapters for MCP tool calling on a local LLM.

**Steps:**

1. Collect or generate ~500-1000 tool-calling traces.
2. Format traces for SFT (chat template format).
3. Train QLoRA adapter on Qwen-2.5-7B-Instruct (requires 1x 16GB+ GPU).
4. Evaluate on held-out tool-calling test set.
5. Optionally: implement decoupled fine-tuning (separate tool selection and argument generation adapters).

**Cost:** ~$5-10 in cloud GPU time (or free if using local hardware).

### Phase 4: DPO Training Loop
- status: todo
- type: task
- id: tunable-layer-guideline.roadmap.phase4
- priority: medium
- estimate: 1d
- blocked_by: [tunable-layer-guideline.roadmap.phase2, tunable-layer-guideline.roadmap.phase3]
<!-- content -->
**Goal:** Close the feedback loop with periodic DPO retraining.

**Steps:**

1. Accumulate ~200+ preference pairs from production logs.
2. Run DPO training on the existing LoRA adapter.
3. Evaluate on held-out preference test set.
4. Deploy updated adapter and monitor quality metrics.
5. Set up a weekly/monthly retraining schedule.

**Cost:** ~$2-5 per training run in cloud GPU time.

## Decision Tree: Which Layer to Implement
- status: active
- type: documentation
- id: tunable-layer-guideline.decision-tree
- last_checked: 2026-02-08
<!-- content -->
Use this decision tree to determine which tunable layer to prioritize based on your deployment configuration:

```
Are you using a cloud LLM API (Gemini, Claude, OpenAI)?
├─ Yes → Can you switch to a local model?
│   ├─ Yes → Implement all three layers (Router → LoRA → DPO)
│   └─ No  → Implement Layer 1 (Router) + Layer 3B (Reranking)
│            Skip Layer 2 (LoRA not possible with cloud APIs)
└─ No (local LLM) → Implement all three layers
    │
    └─ What is your biggest pain point?
        ├─ Wrong retrieval path selected → Start with Layer 1 (Router)
        ├─ LLM calls wrong tools / bad args → Start with Layer 2 (LoRA)
        └─ Answers are okay but not great → Start with Layer 3 (DPO)
```

## Summary: Cost-Benefit Overview
- status: active
- type: documentation
- id: tunable-layer-guideline.summary
- last_checked: 2026-02-08
<!-- content -->

| Layer | Intervention | Effort | GPU Needed | Impact | Works with Cloud API |
|:------|:-------------|:-------|:-----------|:-------|:---------------------|
| **1. Router** | Trained query classifier | 1 day | No (CPU) | High | Yes |
| **2. LoRA** | Tool-calling adapters | 2 days | Yes (16GB+) | High | No (local only) |
| **3a. DPO** | Preference optimization | 1 day | Yes (16GB+) | Medium | No (local only) |
| **3b. Reranking** | Best-of-N with reward model | 1 day | No (CPU) | Medium | Yes |

**Recommendation for the Unified Nexus (cost-sensitive, local-first):**

1. Start with the **Router** (Phase 1) — highest impact, zero GPU cost.
2. Add **Feedback Logging** (Phase 2) — prerequisite for everything else.
3. Add **LoRA** (Phase 3) when a local LLM is deployed.
4. Add **DPO** (Phase 4) once sufficient feedback data accumulates.
