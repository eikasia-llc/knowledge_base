# LinkedIn Profile Knowledge Base Builder
- status: proposed
- type: plan
- owner: antigravity
- context_dependencies: {"conventions": "MD_CONVENTIONS.md", "rag_patterns": "RAGS_AGENT.md", "graph_reference": "mcmp_graph.md"}
- label: ['planning']
<!-- content -->
This document outlines the strategy for building a **slow, respectful LinkedIn profile scraper** that feeds into a personal RAG-based knowledge base with a **Social Graph layer**. The system prioritizes safety (avoiding bans), legal caution (personal use only), and semantic richness (optimized for vector search and graph traversal).

**Philosophy**: Build a "memory" of your professional network over weeks/months, not hours. Quality over speed.

**Key Innovation**: Combines three storage paradigms:
1. **Vector Store (ChromaDB)** — semantic similarity search
2. **Structured Store (DuckDB/JSON)** — precise queries, aggregations
3. **Graph Store (MD + JSON)** — relationship traversal, network analysis

---

## Context & Objectives
- status: active
- type: documentation
- label: ['planning']
<!-- content -->

### Problem Statement
- status: active
- type: documentation
- label: ['planning']
<!-- content -->
LinkedIn's API is locked down for most use cases. We need profile data (skills, experience, about sections) to build a searchable knowledge base of professional connections. Beyond individual profiles, we want to understand **relationships**: who works where, who knows whom, who follows what companies, who engages with whose content.

### Goals
- status: active
- type: documentation
- label: ['planning']
<!-- content -->
1. **Extract** structured profile data from LinkedIn with minimal detection risk.
2. **Transform** raw HTML/data into clean, embeddable documents.
3. **Load** documents into a vector database (ChromaDB) with rich metadata.
4. **Model** relationships between people, companies, and content in a graph.
5. **Search** using hybrid queries: semantic + structured + graph traversal.

### Constraints & Principles
- status: active
- type: documentation
- label: ['planning']
<!-- content -->
- **Personal use only** — never commercial or at scale.
- **Respectful rate limiting** — 30-120+ seconds between requests, daily caps.
- **Human-in-the-loop** — semi-automated, not autonomous scraping.
- **Provenance tracking** — always know where data came from and when.
- **Dual representation** — MD for human readability, JSON for programmatic access.

---

## Architecture Overview
- status: active
- type: documentation
- label: ['planning']
<!-- content -->
```
┌─────────────────────────────────────────────────────────────────┐
│                        LinkedIn Website                         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  COLLECTOR MODULE                                               │
│  ├── SessionManager (cookie-based auth)                         │
│  ├── RateLimiter (configurable delays + daily caps)             │
│  └── ProfileFetcher (Playwright/requests hybrid)                │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  PARSER MODULE                                                  │
│  ├── HTMLParser (BeautifulSoup extraction)                      │
│  ├── ProfileSchema (Pydantic models)                            │
│  ├── Chunker (split profile into embeddable units)              │
│  └── RelationshipExtractor (infer edges from profile data)      │
└─────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
┌───────────────────────────┐  ┌───────────────────────────────────┐
│  STORAGE MODULE           │  │  GRAPH MODULE                     │
│  ├── RawStore (JSON)      │  │  ├── GraphBuilder (nodes/edges)   │
│  ├── ChromaDB (vectors)   │  │  ├── MDGraphWriter (human-read)   │
│  └── DuckDB (structured)  │  │  ├── JSONGraphWriter (programmatic)│
└───────────────────────────┘  │  └── GraphUtils (traversal/query) │
                    │          └───────────────────────────────────┘
                    └───────────┬───────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  SEARCH MODULE (Hybrid)                                         │
│  ├── SemanticSearch (vector similarity)                         │
│  ├── StructuredSearch (DuckDB SQL)                              │
│  ├── GraphSearch (relationship traversal)                       │
│  └── HybridOrchestrator (combines all three)                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  INTERFACE MODULE                                               │
│  ├── CLI (command-line)                                         │
│  ├── MCP Server (tool exposure for LLMs)                        │
│  └── Notebook (interactive exploration)                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure
- status: active
- type: documentation
- label: ['planning']
<!-- content -->
```
linkedin_kb/
├── src/
│   ├── __init__.py
│   ├── config.py               # Settings loader
│   ├── collector/
│   │   ├── __init__.py
│   │   ├── session.py          # Cookie/auth management
│   │   ├── rate_limiter.py     # Delay logic + daily caps
│   │   └── fetcher.py          # Profile page retrieval
│   ├── parser/
│   │   ├── __init__.py
│   │   ├── extractor.py        # HTML -> structured data
│   │   ├── schemas.py          # Pydantic models (profiles, companies, edges)
│   │   ├── chunker.py          # Profile -> embeddable chunks
│   │   └── relationship_extractor.py  # Infer edges from profile data
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── raw_store.py        # JSON file management
│   │   ├── vector_store.py     # ChromaDB operations
│   │   └── structured_store.py # DuckDB operations
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── models.py           # Node/Edge Pydantic models
│   │   ├── builder.py          # Construct graph from profiles
│   │   ├── writers.py          # Export to MD and JSON
│   │   ├── parser.py           # Parse MD/JSON back to graph
│   │   └── utils.py            # Traversal, queries, analysis
│   ├── search/
│   │   ├── __init__.py
│   │   ├── semantic.py         # Vector similarity search
│   │   ├── structured.py       # DuckDB SQL queries
│   │   ├── graph_search.py     # Graph traversal queries
│   │   └── hybrid.py           # Orchestrator combining all
│   ├── mcp/
│   │   ├── __init__.py
│   │   ├── server.py           # MCP server for LLM tools
│   │   └── tools.py            # Tool definitions
│   └── cli.py                  # Command-line interface
├── data/
│   ├── raw/
│   │   ├── profiles/           # Individual profile JSONs
│   │   └── html_cache/         # Raw HTML for re-parsing
│   ├── graph/
│   │   ├── social_graph.md     # Human-readable graph
│   │   ├── social_graph.json   # Programmatic graph
│   │   └── companies.json      # Company entity registry
│   ├── chroma/                 # Vector database
│   ├── duckdb/
│   │   └── linkedin.duckdb     # Structured database
│   └── exports/                # Search results, reports
├── config/
│   ├── settings.yaml           # Rate limits, paths, etc.
│   └── cookies.json            # LinkedIn session (gitignored)
├── notebooks/
│   ├── search_demo.ipynb       # Interactive search examples
│   └── graph_analysis.ipynb    # Network visualization
├── tests/
│   ├── __init__.py
│   ├── test_parser.py
│   ├── test_chunker.py
│   ├── test_graph.py
│   └── test_search.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Implementation Phases
- status: active
- type: plan
- label: ['planning']
<!-- content -->

### Phase 1: Foundation & Configuration
- id: linkedin_kb.phase_1
- status: todo
- type: task
- estimate: 2h
- blocked_by: []
- label: ['planning']
<!-- content -->
Set up project skeleton, configuration system, and basic data models.

#### 1.1 Project Initialization
- id: linkedin_kb.phase_1.init
- status: todo
- type: task
- estimate: 30m
- label: ['planning']
<!-- content -->
**Tasks:**
- [ ] Create directory structure as specified above.
- [ ] Initialize `requirements.txt` with core dependencies.
- [ ] Create `.gitignore` (include `cookies.json`, `data/`, `*.pyc`).
- [ ] Write basic `README.md`.

**Dependencies (requirements.txt):**
```

# Core
- type: plan
- label: ['planning']
<!-- content -->
pydantic>=2.0
pyyaml>=6.0
python-dotenv>=1.0

# Collection
- type: plan
- label: ['planning']
<!-- content -->
playwright>=1.40
requests>=2.31
beautifulsoup4>=4.12
lxml>=5.0

# Storage & Search
- type: plan
- label: ['planning']
<!-- content -->
chromadb>=0.4
sentence-transformers>=2.2
duckdb>=0.9

# Graph
- type: plan
- label: ['planning']
<!-- content -->
networkx>=3.0         # Graph algorithms

# Utilities
- type: plan
- label: ['planning']
<!-- content -->
rich>=13.0            # CLI formatting
tenacity>=8.2         # Retry logic
click>=8.0            # CLI framework
```

#### 1.2 Configuration System
- id: linkedin_kb.phase_1.config
- status: todo
- type: task
- estimate: 45m
- blocked_by: [linkedin_kb.phase_1.init]
- label: ['planning']
<!-- content -->
**Tasks:**
- [ ] Create `config/settings.yaml` with all configurable parameters.
- [ ] Create `src/config.py` to load and validate settings.

**settings.yaml structure:**
```yaml
collector:
  min_delay_seconds: 45
  max_delay_seconds: 120
  daily_request_cap: 30
  user_agents:
    - "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)..."

storage:
  raw_path: "data/raw"
  chroma_path: "data/chroma"
  duckdb_path: "data/duckdb/linkedin.duckdb"
  graph_path: "data/graph"
  collection_name: "linkedin_profiles"

embeddings:
  model: "all-MiniLM-L6-v2"

graph:
  auto_infer_edges: true        # Automatically create edges from profile data
  company_normalization: true   # Normalize company names (Google LLC -> Google)

logging:
  level: "INFO"
  file: "data/scraper.log"
```

#### 1.3 Data Models (Pydantic Schemas)
- id: linkedin_kb.phase_1.schemas
- status: todo
- type: task
- estimate: 45m
- blocked_by: [linkedin_kb.phase_1.init]
- label: ['planning']
<!-- content -->
**Tasks:**
- [ ] Create `src/parser/schemas.py` with profile data models.
- [ ] Create `src/graph/models.py` with node/edge models.

**Profile schemas (src/parser/schemas.py):**
```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Literal

class Experience(BaseModel):
    """Single job/role entry."""
    title: str
    company: str
    company_linkedin_url: Optional[str] = None  # For edge creation
    location: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None  # None = "Present"
    description: Optional[str] = None
    is_current: bool = False
    
class Education(BaseModel):
    """Single education entry."""
    institution: str
    institution_linkedin_url: Optional[str] = None
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    start_year: Optional[int] = None
    end_year: Optional[int] = None

class LinkedInProfile(BaseModel):
    """Complete profile representation."""
    # Identifiers
    profile_url: str
    linkedin_id: Optional[str] = None
    
    # Core info
    name: str
    headline: Optional[str] = None
    location: Optional[str] = None
    about: Optional[str] = None
    
    # Lists
    experience: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    
    # Connections (for graph)
    connection_degree: Optional[int] = None  # 1st, 2nd, 3rd
    
    # Metadata
    collected_at: datetime = Field(default_factory=datetime.utcnow)
    collection_method: str = "manual"
    
class ProfileChunk(BaseModel):
    """Embeddable unit for vector storage."""
    profile_url: str
    chunk_type: Literal["about", "experience", "education", "skills", "summary"]
    content: str
    metadata: dict = Field(default_factory=dict)
```

**Graph schemas (src/graph/models.py):**
```python
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime

# =============================================================================
- type: plan
- label: ['planning']
<!-- content -->

# NODE TYPES
- type: plan
- label: ['planning']
<!-- content -->

# =============================================================================
- type: plan
- label: ['planning']
<!-- content -->
class PersonNode(BaseModel):
    """A person in the network."""
    id: str                          # Slug from profile URL
    name: str
    type: Literal["Person"] = "Person"
    properties: dict = Field(default_factory=dict)
    # properties: headline, location, current_company, skills[]

class CompanyNode(BaseModel):
    """A company/organization."""
    id: str                          # Normalized slug
    name: str
    type: Literal["Company"] = "Company"
    properties: dict = Field(default_factory=dict)
    # properties: industry, size, headquarters, linkedin_url

class InstitutionNode(BaseModel):
    """An educational institution."""
    id: str
    name: str
    type: Literal["Institution"] = "Institution"
    properties: dict = Field(default_factory=dict)

class ContentNode(BaseModel):
    """A piece of content (post, article)."""
    id: str
    title: Optional[str] = None
    type: Literal["Content"] = "Content"
    properties: dict = Field(default_factory=dict)
    # properties: author_id, post_url, created_at

# Union type for all nodes
- type: plan
- label: ['planning']
<!-- content -->
Node = PersonNode | CompanyNode | InstitutionNode | ContentNode

# =============================================================================
- type: plan
- label: ['planning']
<!-- content -->

# EDGE TYPES (Relationships)
- type: plan
- label: ['planning']
<!-- content -->

# =============================================================================
- type: plan
- label: ['planning']
<!-- content -->
class Edge(BaseModel):
    """A relationship between two nodes."""
    source: str                      # Source node ID
    target: str                      # Target node ID
    relationship: str                # Relationship type
    properties: dict = Field(default_factory=dict)
    # Common properties: start_date, end_date, role, observed_at

# Relationship type constants
- type: plan
- label: ['planning']
<!-- content -->
class RelationshipTypes:
    # Person <-> Company
    WORKS_AT = "works_at"            # Current employment
    WORKED_AT = "worked_at"          # Past employment
    FOLLOWS = "follows"              # Following company page
    
    # Person <-> Person
    CONNECTED_TO = "connected_to"    # 1st degree connection
    FOLLOWS_PERSON = "follows"       # Following (asymmetric)
    LIKED_POST_BY = "liked_post_by"  # Engagement signal
    COMMENTED_ON = "commented_on"    # Engagement signal
    COLLEAGUE_OF = "colleague_of"    # Worked at same company (inferred)
    CLASSMATE_OF = "classmate_of"    # Same institution (inferred)
    
    # Person <-> Institution
    STUDIED_AT = "studied_at"
    
    # Person <-> Content
    AUTHORED = "authored"
    LIKED = "liked"
    SHARED = "shared"

# =============================================================================
- type: plan
- label: ['planning']
<!-- content -->

# GRAPH CONTAINER
- type: plan
- label: ['planning']
<!-- content -->

# =============================================================================
- type: plan
- label: ['planning']
<!-- content -->
class SocialGraph(BaseModel):
    """Complete graph representation."""
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    # metadata: last_updated, node_count, edge_count
    
    def add_node(self, node: Node) -> None:
        """Add node if not exists."""
        if not any(n.id == node.id for n in self.nodes):
            self.nodes.append(node)
    
    def add_edge(self, edge: Edge) -> None:
        """Add edge if not exists."""
        exists = any(
            e.source == edge.source and 
            e.target == edge.target and 
            e.relationship == edge.relationship
            for e in self.edges
        )
        if not exists:
            self.edges.append(edge)
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get node by ID."""
        return next((n for n in self.nodes if n.id == node_id), None)
    
    def get_edges_for_node(self, node_id: str) -> list[Edge]:
        """Get all edges involving a node."""
        return [e for e in self.edges if e.source == node_id or e.target == node_id]
```

---

### Phase 2: Collection Module
- id: linkedin_kb.phase_2
- status: todo
- type: task
- estimate: 4h
- blocked_by: [linkedin_kb.phase_1]
- label: ['planning']
<!-- content -->
Build the safe, rate-limited profile fetching system.

#### 2.1 Session Manager
- id: linkedin_kb.phase_2.session
- status: todo
- type: task
- estimate: 1h
- label: ['planning']
<!-- content -->
**Tasks:**
- [ ] Create `src/collector/session.py`.
- [ ] Implement cookie loading from JSON file.
- [ ] Implement session validation (check if still logged in).
- [ ] Add user-agent rotation.

**Cookie extraction process (manual, one-time):**
```python

# 1. Log into LinkedIn in Chrome
- type: plan
- label: ['planning']
<!-- content -->

# 2. Open DevTools > Application > Cookies
- type: plan
- label: ['planning']
<!-- content -->

# 3. Export li_at and JSESSIONID cookies
- type: plan
- label: ['planning']
<!-- content -->

# 4. Save to config/cookies.json (gitignored!)
- type: plan
- label: ['planning']
<!-- content -->

# cookies.json format:
- type: plan
- label: ['planning']
<!-- content -->
{
    "li_at": "AQED...",
    "JSESSIONID": "ajax:123..."
}
```

#### 2.2 Rate Limiter
- id: linkedin_kb.phase_2.rate_limiter
- status: todo
- type: task
- estimate: 1h
- blocked_by: [linkedin_kb.phase_2.session]
- label: ['planning']
<!-- content -->
**Tasks:**
- [ ] Create `src/collector/rate_limiter.py`.
- [ ] Implement random delay between requests (configurable range).
- [ ] Implement daily request counter with persistent state.
- [ ] Add "cooldown" mode if approaching daily cap.

**Key implementation:**
```python
import random
import time
import json
from pathlib import Path
from datetime import date

class RateLimiter:
    """
    Enforce rate limits to avoid LinkedIn detection.
    
    Features:
    - Random delays between requests (human-like)
    - Daily request caps with persistent tracking
    - Occasional "break" delays (mimics human behavior)
    """
    
    def __init__(self, config):
        self.min_delay = config.min_delay_seconds
        self.max_delay = config.max_delay_seconds
        self.daily_cap = config.daily_request_cap
        self.state_file = Path("data/.rate_limiter_state.json")
    
    def wait(self) -> float:
        """
        Block for random delay. Call before each request.
        
        Returns:
            float: Actual delay in seconds
        """
        delay = random.uniform(self.min_delay, self.max_delay)
        
        # 10% chance of a longer "break" (mimics human behavior)
        if random.random() < 0.1:
            delay *= random.uniform(2, 5)
        
        time.sleep(delay)
        return delay
    
    def can_request(self) -> bool:
        """Check if under daily cap."""
        today_count = self._load_today_count()
        return today_count < self.daily_cap
    
    def record_request(self) -> int:
        """
        Increment today's counter.
        
        Returns:
            int: Updated count for today
        """
        state = self._load_state()
        today = str(date.today())
        state[today] = state.get(today, 0) + 1
        self._save_state(state)
        return state[today]
    
    def remaining_today(self) -> int:
        """Get remaining requests for today."""
        return max(0, self.daily_cap - self._load_today_count())
    
    def _load_today_count(self) -> int:
        state = self._load_state()
        return state.get(str(date.today()), 0)
    
    def _load_state(self) -> dict:
        if self.state_file.exists():
            return json.loads(self.state_file.read_text())
        return {}
    
    def _save_state(self, state: dict) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(state, indent=2))
```

#### 2.3 Profile Fetcher
- id: linkedin_kb.phase_2.fetcher
- status: todo
- type: task
- estimate: 2h
- blocked_by: [linkedin_kb.phase_2.rate_limiter]
- label: ['planning']
<!-- content -->
**Tasks:**
- [ ] Create `src/collector/fetcher.py`.
- [ ] Implement Playwright-based fetcher (handles JS-rendered content).
- [ ] Add retry logic with exponential backoff.
- [ ] Detect and handle soft blocks (CAPTCHAs, rate limit pages).

**Soft block detection:**
- Check for "unusual activity" page.
- Check for CAPTCHA elements.
- Check for login redirect.
- If detected: log, pause for hours, alert user.

---

### Phase 3: Parser Module
- id: linkedin_kb.phase_3
- status: todo
- type: task
- estimate: 3h
- blocked_by: [linkedin_kb.phase_2]
- label: ['planning']
<!-- content -->
Extract structured data from raw HTML and prepare for embedding.

#### 3.1 HTML Extractor
- id: linkedin_kb.phase_3.extractor
- status: todo
- type: task
- estimate: 1.5h
- label: ['planning']
<!-- content -->
**Tasks:**
- [ ] Create `src/parser/extractor.py`.
- [ ] Implement extraction for each profile section.
- [ ] Handle missing sections gracefully.
- [ ] Add fallback selectors (LinkedIn changes HTML frequently).
- [ ] Extract company/institution URLs for graph edges.

**Key addition for graph support:**
```python
def extract_experience(soup) -> list[Experience]:
    """
    Extract experience with company links for graph edges.
    """
    experiences = []
    for item in soup.select(EXPERIENCE_SELECTORS):
        exp = Experience(
            title=extract_text(item, TITLE_SELECTORS),
            company=extract_text(item, COMPANY_SELECTORS),
            # NEW: Extract company page URL for edge creation
            company_linkedin_url=extract_href(item, COMPANY_LINK_SELECTORS),
            # ... rest of extraction
        )
        experiences.append(exp)
    return experiences
```

#### 3.2 Profile Chunker
- id: linkedin_kb.phase_3.chunker
- status: todo
- type: task
- estimate: 1h
- blocked_by: [linkedin_kb.phase_3.extractor]
- label: ['planning']
<!-- content -->
**Tasks:**
- [ ] Create `src/parser/chunker.py`.
- [ ] Implement chunking strategies for each profile section.
- [ ] Generate synthetic summary chunk.
- [ ] Ensure chunks have rich metadata for filtering.

#### 3.3 Relationship Extractor
- id: linkedin_kb.phase_3.relationships
- status: todo
- type: task
- estimate: 30m
- blocked_by: [linkedin_kb.phase_3.extractor]
- label: ['planning']
<!-- content -->
**Tasks:**
- [ ] Create `src/parser/relationship_extractor.py`.
- [ ] Extract implicit edges from profile data.
- [ ] Normalize company/institution names.

**Implementation:**
```python
from src.graph.models import Edge, RelationshipTypes, CompanyNode
from src.parser.schemas import LinkedInProfile
import re

def slugify(name: str) -> str:
    """Convert name to URL-safe slug."""
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')

def extract_edges_from_profile(profile: LinkedInProfile) -> tuple[list[Edge], list[CompanyNode]]:
    """
    Extract graph edges from a profile.
    
    Returns:
        Tuple of (edges, company_nodes_to_create)
    """
    edges = []
    companies = []
    person_id = slugify(profile.name)
    
    for exp in profile.experience:
        company_id = slugify(exp.company)
        
        # Create company node reference
        companies.append(CompanyNode(
            id=company_id,
            name=exp.company,
            properties={"linkedin_url": exp.company_linkedin_url}
        ))
        
        # Create employment edge
        relationship = RelationshipTypes.WORKS_AT if exp.is_current else RelationshipTypes.WORKED_AT
        edges.append(Edge(
            source=person_id,
            target=company_id,
            relationship=relationship,
            properties={
                "role": exp.title,
                "start_date": exp.start_date,
                "end_date": exp.end_date,
                "location": exp.location,
            }
        ))
    
    for edu in profile.education:
        institution_id = slugify(edu.institution)
        edges.append(Edge(
            source=person_id,
            target=institution_id,
            relationship=RelationshipTypes.STUDIED_AT,
            properties={
                "degree": edu.degree,
                "field": edu.field_of_study,
                "start_year": edu.start_year,
                "end_year": edu.end_year,
            }
        ))
    
    return edges, companies
```

---

### Phase 4: Storage Module
- id: linkedin_kb.phase_4
- status: todo
- type: task
- estimate: 3h
- blocked_by: [linkedin_kb.phase_3]
- label: ['planning']
<!-- content -->
Persist raw data and create searchable vector embeddings.

#### 4.1 Raw JSON Store
- id: linkedin_kb.phase_4.raw_store
- status: todo
- type: task
- estimate: 45m
- label: ['planning']
<!-- content -->
**Tasks:**
- [ ] Create `src/storage/raw_store.py`.
- [ ] Implement save/load for individual profiles.
- [ ] Implement deduplication (by profile URL).
- [ ] Add backup/versioning for re-parsing later.

#### 4.2 Vector Store (ChromaDB)
- id: linkedin_kb.phase_4.vector_store
- status: todo
- type: task
- estimate: 1.5h
- blocked_by: [linkedin_kb.phase_4.raw_store]
- label: ['planning']
<!-- content -->
**Tasks:**
- [ ] Create `src/storage/vector_store.py`.
- [ ] Initialize ChromaDB with persistent storage.
- [ ] Implement batch embedding and insertion.
- [ ] Implement upsert logic (update existing profiles).

#### 4.3 Structured Store (DuckDB)
- id: linkedin_kb.phase_4.structured_store
- status: todo
- type: task
- estimate: 45m
- blocked_by: [linkedin_kb.phase_4.raw_store]
- label: ['planning']
<!-- content -->
**Tasks:**
- [ ] Create `src/storage/structured_store.py`.
- [ ] Define schema for profiles, experiences, skills.
- [ ] Implement insert/update operations.
- [ ] Add common query patterns.

**DuckDB Schema:**
```sql
-- Profiles table
CREATE TABLE IF NOT EXISTS profiles (
    profile_url VARCHAR PRIMARY KEY,
    linkedin_id VARCHAR,
    name VARCHAR NOT NULL,
    headline VARCHAR,
    location VARCHAR,
    about TEXT,
    connection_degree INTEGER,
    collected_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Experiences table (for precise queries)
CREATE TABLE IF NOT EXISTS experiences (
    id INTEGER PRIMARY KEY,
    profile_url VARCHAR REFERENCES profiles(profile_url),
    title VARCHAR,
    company VARCHAR,
    company_id VARCHAR,  -- Links to graph
    location VARCHAR,
    start_date VARCHAR,
    end_date VARCHAR,
    is_current BOOLEAN,
    description TEXT
);

-- Skills table
CREATE TABLE IF NOT EXISTS skills (
    profile_url VARCHAR REFERENCES profiles(profile_url),
    skill VARCHAR,
    PRIMARY KEY (profile_url, skill)
);

-- Useful views
CREATE VIEW current_employees AS
SELECT p.name, p.headline, e.company, e.title, e.location
FROM profiles p
JOIN experiences e ON p.profile_url = e.profile_url
WHERE e.is_current = true;
```

---

### Phase 5: Search Module
- id: linkedin_kb.phase_5
- status: todo
- type: task
- estimate: 2h
- blocked_by: [linkedin_kb.phase_4]
- label: ['planning']
<!-- content -->
Enable semantic queries across the knowledge base.

#### 5.1 Semantic Search
- id: linkedin_kb.phase_5.semantic
- status: todo
- type: task
- estimate: 1h
- label: ['planning']
<!-- content -->
**Tasks:**
- [ ] Create `src/search/semantic.py`.
- [ ] Implement basic similarity search.
- [ ] Implement filtered search (by company, location, etc.).
- [ ] Add result grouping (aggregate chunks by profile).

#### 5.2 Structured Search
- id: linkedin_kb.phase_5.structured
- status: todo
- type: task
- estimate: 30m
- label: ['planning']
<!-- content -->
**Tasks:**
- [ ] Create `src/search/structured.py`.
- [ ] Implement common query patterns via DuckDB.

**Query patterns:**
```python
class StructuredSearch:
    """SQL-based precise queries."""
    
    def people_at_company(self, company: str, current_only: bool = True) -> list[dict]:
        """Find all people who work(ed) at a company."""
        query = """
            SELECT DISTINCT p.name, p.headline, e.title, e.is_current
            FROM profiles p
            JOIN experiences e ON p.profile_url = e.profile_url
            WHERE LOWER(e.company) LIKE LOWER(?)
        """
        if current_only:
            query += " AND e.is_current = true"
        return self.db.execute(query, [f"%{company}%"]).fetchall()
    
    def people_with_skill(self, skill: str) -> list[dict]:
        """Find all people with a specific skill."""
        query = """
            SELECT p.name, p.headline, p.profile_url
            FROM profiles p
            JOIN skills s ON p.profile_url = s.profile_url
            WHERE LOWER(s.skill) LIKE LOWER(?)
        """
        return self.db.execute(query, [f"%{skill}%"]).fetchall()
    
    def company_headcount(self) -> list[dict]:
        """Count connections by company."""
        query = """
            SELECT e.company, COUNT(DISTINCT p.profile_url) as headcount
            FROM profiles p
            JOIN experiences e ON p.profile_url = e.profile_url
            WHERE e.is_current = true
            GROUP BY e.company
            ORDER BY headcount DESC
        """
        return self.db.execute(query).fetchall()
```

#### 5.3 CLI Interface
- id: linkedin_kb.phase_5.cli
- status: todo
- type: task
- estimate: 30m
- blocked_by: [linkedin_kb.phase_5.semantic]
- label: ['planning']
<!-- content -->
**Tasks:**
- [ ] Create `src/cli.py` using `click`.
- [ ] Implement commands: `collect`, `search`, `stats`, `export`, `graph`.

---

### Phase 6: Testing & Documentation
- id: linkedin_kb.phase_6
- status: todo
- type: task
- estimate: 2h
- blocked_by: [linkedin_kb.phase_5]
- label: ['planning']
<!-- content -->
Ensure reliability and usability.

#### 6.1 Unit Tests
- id: linkedin_kb.phase_6.tests
- status: todo
- type: task
- estimate: 1.5h
- label: ['planning']
<!-- content -->
**Tasks:**
- [ ] Create `tests/test_parser.py` — test extraction with sample HTML.
- [ ] Create `tests/test_chunker.py` — test chunking logic.
- [ ] Create `tests/test_graph.py` — test graph building and queries.
- [ ] Create `tests/test_search.py` — test search with mock data.

#### 6.2 Documentation
- id: linkedin_kb.phase_6.docs
- status: todo
- type: task
- estimate: 30m
- blocked_by: [linkedin_kb.phase_6.tests]
- label: ['planning']
<!-- content -->
**Tasks:**
- [ ] Write comprehensive `README.md`.
- [ ] Document cookie extraction process step-by-step.
- [ ] Add usage examples in notebooks.
- [ ] Include legal/ethical disclaimer.

---

### Phase 7: Social Graph Module
- id: linkedin_kb.phase_7
- status: todo
- type: task
- estimate: 5h
- blocked_by: [linkedin_kb.phase_4]
- label: ['planning']
<!-- content -->
Build the relationship graph layer following MCMP patterns. This is the key differentiator that enables network-level queries.

#### 7.1 Graph Builder
- id: linkedin_kb.phase_7.builder
- status: todo
- type: task
- estimate: 1.5h
- label: ['planning']
<!-- content -->
**Tasks:**
- [ ] Create `src/graph/builder.py`.
- [ ] Implement profile-to-graph conversion.
- [ ] Implement incremental graph updates.
- [ ] Infer colleague/classmate relationships.

**Implementation:**
```python
from src.graph.models import SocialGraph, PersonNode, CompanyNode, Edge, RelationshipTypes
from src.parser.schemas import LinkedInProfile
from src.parser.relationship_extractor import extract_edges_from_profile, slugify
from typing import Optional

class GraphBuilder:
    """
    Build and maintain the social graph from profile data.
    
    Follows MCMP pattern: dual MD + JSON representation.
    """
    
    def __init__(self, graph: Optional[SocialGraph] = None):
        self.graph = graph or SocialGraph()
    
    def add_profile(self, profile: LinkedInProfile) -> None:
        """
        Add a profile to the graph.
        
        Creates:
        - Person node
        - Company/Institution nodes
        - Employment/Education edges
        """
        person_id = slugify(profile.name)
        
        # Create person node
        person_node = PersonNode(
            id=person_id,
            name=profile.name,
            properties={
                "headline": profile.headline,
                "location": profile.location,
                "profile_url": profile.profile_url,
                "skills": profile.skills[:10],
                "collected_at": profile.collected_at.isoformat(),
            }
        )
        self.graph.add_node(person_node)
        
        # Extract and add edges
        edges, companies = extract_edges_from_profile(profile)
        
        for company in companies:
            self.graph.add_node(company)
        
        for edge in edges:
            self.graph.add_edge(edge)
    
    def infer_colleague_edges(self) -> int:
        """
        Infer colleague relationships from overlapping employment.
        
        Two people are colleagues if they worked at the same company
        during overlapping time periods.
        
        Returns:
            Number of new edges created
        """
        # Group people by company
        company_employees: dict[str, list[tuple[str, Edge]]] = {}
        
        for edge in self.graph.edges:
            if edge.relationship in [RelationshipTypes.WORKS_AT, RelationshipTypes.WORKED_AT]:
                company_id = edge.target
                if company_id not in company_employees:
                    company_employees[company_id] = []
                company_employees[company_id].append((edge.source, edge))
        
        # Create colleague edges for overlapping employment
        new_edges = 0
        for company_id, employees in company_employees.items():
            for i, (person_a, edge_a) in enumerate(employees):
                for person_b, edge_b in employees[i+1:]:
                    if self._periods_overlap(edge_a.properties, edge_b.properties):
                        colleague_edge = Edge(
                            source=person_a,
                            target=person_b,
                            relationship=RelationshipTypes.COLLEAGUE_OF,
                            properties={
                                "company": company_id,
                                "inferred": True,
                            }
                        )
                        self.graph.add_edge(colleague_edge)
                        new_edges += 1
        
        return new_edges
    
    def _periods_overlap(self, props_a: dict, props_b: dict) -> bool:
        """Check if two employment periods overlap."""
        # Simplified: if either is current, they overlap
        # More sophisticated: parse dates and check ranges
        if props_a.get("end_date") is None or props_b.get("end_date") is None:
            return True
        # TODO: Implement proper date overlap checking
        return True
```

#### 7.2 Graph Writers (MD + JSON)
- id: linkedin_kb.phase_7.writers
- status: todo
- type: task
- estimate: 1.5h
- blocked_by: [linkedin_kb.phase_7.builder]
- label: ['planning']
<!-- content -->
**Tasks:**
- [ ] Create `src/graph/writers.py`.
- [ ] Implement Markdown writer (human-readable, MCMP style).
- [ ] Implement JSON writer (programmatic access).
- [ ] Ensure lossless round-trip conversion.

**Markdown format (following MCMP pattern):**
```markdown

# LinkedIn Social Graph
- status: active
- type: documentation
- context_dependencies: {'utils': 'src/graph/utils.py'}
- label: ['planning']
<!-- content -->

### Nodes
- type: task
- label: ['planning']
<!-- content -->
| id | name | type | properties |
|---|---|---|---|
| john_doe | John Doe | Person | headline: ML Engineer at Google, location: Bay Area |
| jane_smith | Jane Smith | Person | headline: PM at Meta, location: NYC |
| google | Google | Company | industry: Tech |
| meta | Meta | Company | industry: Tech |
| stanford | Stanford University | Institution | type: University |

### Edges
- type: task
- label: ['planning']
<!-- content -->
| source | target | relationship | properties |
|---|---|---|---|
| john_doe | google | works_at | role: ML Engineer, start_date: 2022-01 |
| john_doe | stanford | studied_at | degree: MS, field: Computer Science |
| jane_smith | meta | works_at | role: Product Manager |
| john_doe | jane_smith | colleague_of | company: google, inferred: true |
```

**Implementation:**
```python
from src.graph.models import SocialGraph, Node, Edge
from pathlib import Path
import json

class GraphWriter:
    """Write graph to MD and JSON formats."""
    
    @staticmethod
    def to_markdown(graph: SocialGraph, path: Path) -> None:
        """
        Write graph to Markdown following MD_CONVENTIONS.
        """
        lines = [
            "# LinkedIn Social Graph",
            "- status: active",
            "- type: context",
            "- context_dependencies: {'utils': 'src/graph/utils.py'}",
            "<!-- content -->",
            "",
            "### Nodes",
            "| id | name | type | properties |",
            "|---|---|---|---|",
        ]
        
        for node in graph.nodes:
            props_str = ", ".join(f"{k}: {v}" for k, v in node.properties.items() if v)
            lines.append(f"| {node.id} | {node.name} | {node.type} | {props_str} |")
        
        lines.extend([
            "",
            "### Edges",
            "| source | target | relationship | properties |",
            "|---|---|---|---|",
        ])
        
        for edge in graph.edges:
            props_str = ", ".join(f"{k}: {v}" for k, v in edge.properties.items() if v)
            lines.append(f"| {edge.source} | {edge.target} | {edge.relationship} | {props_str} |")
        
        path.write_text("\n".join(lines))
    
    @staticmethod
    def to_json(graph: SocialGraph, path: Path) -> None:
        """
        Write graph to JSON for programmatic access.
        """
        data = {
            "nodes": [n.model_dump() for n in graph.nodes],
            "edges": [e.model_dump() for e in graph.edges],
            "metadata": graph.metadata,
        }
        path.write_text(json.dumps(data, indent=4, default=str))
```

#### 7.3 Graph Parser
- id: linkedin_kb.phase_7.parser
- status: todo
- type: task
- estimate: 1h
- blocked_by: [linkedin_kb.phase_7.writers]
- label: ['planning']
<!-- content -->
**Tasks:**
- [ ] Create `src/graph/parser.py`.
- [ ] Implement MD parser (read tables back to graph).
- [ ] Implement JSON parser.
- [ ] Validate round-trip consistency.

**Implementation:**
```python
from src.graph.models import SocialGraph, PersonNode, CompanyNode, InstitutionNode, Edge
from pathlib import Path
import json
import re

class GraphParser:
    """Parse graph from MD and JSON formats."""
    
    @staticmethod
    def from_json(path: Path) -> SocialGraph:
        """Load graph from JSON file."""
        data = json.loads(path.read_text())
        
        nodes = []
        for n in data["nodes"]:
            node_type = n["type"]
            if node_type == "Person":
                nodes.append(PersonNode(**n))
            elif node_type == "Company":
                nodes.append(CompanyNode(**n))
            elif node_type == "Institution":
                nodes.append(InstitutionNode(**n))
        
        edges = [Edge(**e) for e in data["edges"]]
        
        return SocialGraph(
            nodes=nodes,
            edges=edges,
            metadata=data.get("metadata", {})
        )
    
    @staticmethod
    def from_markdown(path: Path) -> SocialGraph:
        """
        Parse graph from Markdown table format.
        """
        content = path.read_text()
        graph = SocialGraph()
        
        # Parse nodes table
        nodes_match = re.search(
            r"### Nodes\n\|.*\|\n\|[-|]+\|\n((?:\|.*\|\n)*)",
            content
        )
        if nodes_match:
            for line in nodes_match.group(1).strip().split("\n"):
                parts = [p.strip() for p in line.strip("|").split("|")]
                if len(parts) >= 4:
                    node_id, name, node_type, props_str = parts[:4]
                    properties = GraphParser._parse_properties(props_str)
                    
                    if node_type == "Person":
                        graph.add_node(PersonNode(id=node_id, name=name, properties=properties))
                    elif node_type == "Company":
                        graph.add_node(CompanyNode(id=node_id, name=name, properties=properties))
                    elif node_type == "Institution":
                        graph.add_node(InstitutionNode(id=node_id, name=name, properties=properties))
        
        # Parse edges table
        edges_match = re.search(
            r"### Edges\n\|.*\|\n\|[-|]+\|\n((?:\|.*\|\n)*)",
            content
        )
        if edges_match:
            for line in edges_match.group(1).strip().split("\n"):
                parts = [p.strip() for p in line.strip("|").split("|")]
                if len(parts) >= 4:
                    source, target, relationship, props_str = parts[:4]
                    properties = GraphParser._parse_properties(props_str)
                    graph.add_edge(Edge(
                        source=source,
                        target=target,
                        relationship=relationship,
                        properties=properties
                    ))
        
        return graph
    
    @staticmethod
    def _parse_properties(props_str: str) -> dict:
        """Parse 'key: value, key2: value2' format."""
        if not props_str.strip():
            return {}
        props = {}
        for pair in props_str.split(","):
            if ":" in pair:
                key, value = pair.split(":", 1)
                props[key.strip()] = value.strip()
        return props
```

#### 7.4 Graph Utilities & Queries
- id: linkedin_kb.phase_7.utils
- status: todo
- type: task
- estimate: 1h
- blocked_by: [linkedin_kb.phase_7.parser]
- label: ['planning']
<!-- content -->
**Tasks:**
- [ ] Create `src/graph/utils.py`.
- [ ] Implement common traversal patterns.
- [ ] Implement network analysis queries.
- [ ] Support integration with NetworkX for advanced analysis.

**Implementation:**
```python
from src.graph.models import SocialGraph, Node, Edge, RelationshipTypes
from typing import Optional
import networkx as nx

class GraphUtils:
    """
    Graph traversal and analysis utilities.
    """
    
    def __init__(self, graph: SocialGraph):
        self.graph = graph
        self._nx_graph: Optional[nx.DiGraph] = None
    
    # =========================================================================
    # BASIC QUERIES
    # =========================================================================
    
    def get_people_at_company(self, company_id: str, current_only: bool = True) -> list[Node]:
        """
        Get all people who work(ed) at a company.
        
        Args:
            company_id: Company node ID
            current_only: If True, only return current employees
        
        Returns:
            List of Person nodes
        """
        relationships = [RelationshipTypes.WORKS_AT]
        if not current_only:
            relationships.append(RelationshipTypes.WORKED_AT)
        
        person_ids = set()
        for edge in self.graph.edges:
            if edge.target == company_id and edge.relationship in relationships:
                person_ids.add(edge.source)
        
        return [self.graph.get_node(pid) for pid in person_ids if self.graph.get_node(pid)]
    
    def get_companies_for_person(self, person_id: str) -> list[tuple[Node, Edge]]:
        """
        Get all companies a person has worked at.
        
        Returns:
            List of (company_node, employment_edge) tuples
        """
        results = []
        for edge in self.graph.edges:
            if edge.source == person_id and edge.relationship in [
                RelationshipTypes.WORKS_AT, RelationshipTypes.WORKED_AT
            ]:
                company = self.graph.get_node(edge.target)
                if company:
                    results.append((company, edge))
        return results
    
    def get_colleagues(self, person_id: str) -> list[Node]:
        """
        Get all colleagues (inferred or explicit).
        """
        colleague_ids = set()
        for edge in self.graph.edges:
            if edge.relationship == RelationshipTypes.COLLEAGUE_OF:
                if edge.source == person_id:
                    colleague_ids.add(edge.target)
                elif edge.target == person_id:
                    colleague_ids.add(edge.source)
        
        return [self.graph.get_node(cid) for cid in colleague_ids if self.graph.get_node(cid)]
    
    def get_network_for_company(self, company_id: str) -> list[Node]:
        """
        Get all people connected to a company (employees, followers, etc.)
        """
        person_ids = set()
        for edge in self.graph.edges:
            if edge.target == company_id:
                person_ids.add(edge.source)
        
        return [self.graph.get_node(pid) for pid in person_ids if self.graph.get_node(pid)]
    
    # =========================================================================
    # NETWORK ANALYSIS (via NetworkX)
    # =========================================================================
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX graph for advanced analysis."""
        if self._nx_graph is None:
            G = nx.DiGraph()
            
            for node in self.graph.nodes:
                G.add_node(node.id, **{"name": node.name, "type": node.type, **node.properties})
            
            for edge in self.graph.edges:
                G.add_edge(edge.source, edge.target, relationship=edge.relationship, **edge.properties)
            
            self._nx_graph = G
        return self._nx_graph
    
    def find_path(self, source_id: str, target_id: str) -> list[str]:
        """
        Find shortest path between two nodes.
        
        Returns:
            List of node IDs representing the path
        """
        G = self.to_networkx()
        try:
            return nx.shortest_path(G.to_undirected(), source_id, target_id)
        except nx.NetworkXNoPath:
            return []
    
    def get_mutual_connections(self, person_a: str, person_b: str) -> list[Node]:
        """
        Find people connected to both person_a and person_b.
        """
        connections_a = set(self._get_connected_people(person_a))
        connections_b = set(self._get_connected_people(person_b))
        mutual = connections_a & connections_b
        return [self.graph.get_node(pid) for pid in mutual if self.graph.get_node(pid)]
    
    def _get_connected_people(self, person_id: str) -> set[str]:
        """Get all person IDs directly connected to a person."""
        connected = set()
        for edge in self.graph.edges:
            if edge.source == person_id:
                target_node = self.graph.get_node(edge.target)
                if target_node and target_node.type == "Person":
                    connected.add(edge.target)
            elif edge.target == person_id:
                source_node = self.graph.get_node(edge.source)
                if source_node and source_node.type == "Person":
                    connected.add(edge.source)
        return connected
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    def get_stats(self) -> dict:
        """Get graph statistics."""
        G = self.to_networkx()
        
        node_counts = {}
        for node in self.graph.nodes:
            node_counts[node.type] = node_counts.get(node.type, 0) + 1
        
        edge_counts = {}
        for edge in self.graph.edges:
            edge_counts[edge.relationship] = edge_counts.get(edge.relationship, 0) + 1
        
        return {
            "total_nodes": len(self.graph.nodes),
            "total_edges": len(self.graph.edges),
            "nodes_by_type": node_counts,
            "edges_by_type": edge_counts,
            "density": nx.density(G) if len(G.nodes) > 1 else 0,
        }
```

---

### Phase 8: Hybrid Search & MCP Integration
- id: linkedin_kb.phase_8
- status: todo
- type: task
- estimate: 3h
- blocked_by: [linkedin_kb.phase_7, linkedin_kb.phase_5]
- label: ['planning']
<!-- content -->
Combine all search modalities and expose as MCP tools.

#### 8.1 Hybrid Search Orchestrator
- id: linkedin_kb.phase_8.hybrid
- status: todo
- type: task
- estimate: 1.5h
- label: ['planning']
<!-- content -->
**Tasks:**
- [ ] Create `src/search/hybrid.py`.
- [ ] Implement query routing logic.
- [ ] Combine results from semantic, structured, and graph searches.

**Implementation:**
```python
from src.search.semantic import SemanticSearch
from src.search.structured import StructuredSearch
from src.search.graph_search import GraphSearch
from dataclasses import dataclass
from enum import Enum

class QueryType(Enum):
    SEMANTIC = "semantic"       # "ML engineers with startup experience"
    STRUCTURED = "structured"   # "Who works at Google?"
    GRAPH = "graph"             # "Who are John's colleagues?"
    HYBRID = "hybrid"           # Complex queries needing multiple sources

@dataclass
class HybridResult:
    """Result from hybrid search."""
    query: str
    query_type: QueryType
    results: list[dict]
    sources_used: list[str]

class HybridOrchestrator:
    """
    Route queries to appropriate search backends and combine results.
    """
    
    def __init__(self, semantic: SemanticSearch, structured: StructuredSearch, graph: GraphSearch):
        self.semantic = semantic
        self.structured = structured
        self.graph = graph
    
    def search(self, query: str, mode: str = "auto") -> HybridResult:
        """
        Execute search with automatic or specified mode.
        
        Args:
            query: Natural language query
            mode: "auto", "semantic", "structured", "graph", or "hybrid"
        
        Returns:
            HybridResult with combined results
        """
        if mode == "auto":
            query_type = self._classify_query(query)
        else:
            query_type = QueryType(mode)
        
        sources_used = []
        results = []
        
        if query_type == QueryType.SEMANTIC:
            results = self.semantic.search(query)
            sources_used = ["vector_store"]
        
        elif query_type == QueryType.STRUCTURED:
            results = self._execute_structured_query(query)
            sources_used = ["duckdb"]
        
        elif query_type == QueryType.GRAPH:
            results = self._execute_graph_query(query)
            sources_used = ["graph"]
        
        elif query_type == QueryType.HYBRID:
            # Execute all and merge
            semantic_results = self.semantic.search(query)
            structured_results = self._execute_structured_query(query)
            graph_results = self._execute_graph_query(query)
            
            results = self._merge_results(semantic_results, structured_results, graph_results)
            sources_used = ["vector_store", "duckdb", "graph"]
        
        return HybridResult(
            query=query,
            query_type=query_type,
            results=results,
            sources_used=sources_used
        )
    
    def _classify_query(self, query: str) -> QueryType:
        """
        Classify query type based on patterns.
        """
        query_lower = query.lower()
        
        # Graph patterns
        graph_keywords = ["colleagues", "connections", "path to", "mutual", "network of"]
        if any(kw in query_lower for kw in graph_keywords):
            return QueryType.GRAPH
        
        # Structured patterns
        structured_keywords = ["who works at", "people at", "list all", "count", "how many"]
        if any(kw in query_lower for kw in structured_keywords):
            return QueryType.STRUCTURED
        
        # Default to semantic for open-ended queries
        return QueryType.SEMANTIC
    
    def _execute_structured_query(self, query: str) -> list[dict]:
        """Parse and execute structured query."""
        # Simple pattern matching for common queries
        query_lower = query.lower()
        
        if "who works at" in query_lower or "people at" in query_lower:
            # Extract company name
            import re
            match = re.search(r"(?:who works at|people at)\s+(.+?)(?:\?|$)", query_lower)
            if match:
                company = match.group(1).strip()
                return self.structured.people_at_company(company)
        
        return []
    
    def _execute_graph_query(self, query: str) -> list[dict]:
        """Parse and execute graph query."""
        # TODO: Implement graph query parsing
        return []
    
    def _merge_results(self, *result_sets) -> list[dict]:
        """Merge and deduplicate results from multiple sources."""
        seen_urls = set()
        merged = []
        
        for results in result_sets:
            for r in results:
                url = r.get("profile_url") or r.get("url")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    merged.append(r)
        
        return merged
```

#### 8.2 MCP Server
- id: linkedin_kb.phase_8.mcp
- status: todo
- type: task
- estimate: 1.5h
- blocked_by: [linkedin_kb.phase_8.hybrid]
- label: ['planning']
<!-- content -->
**Tasks:**
- [ ] Create `src/mcp/server.py`.
- [ ] Create `src/mcp/tools.py`.
- [ ] Expose search functions as MCP tools.
- [ ] Enable LLM integration.

**Tool definitions:**
```python
from mcp import FastMCP
from src.search.hybrid import HybridOrchestrator

mcp = FastMCP("linkedin_kb")

@mcp.tool()
def search_profiles(query: str, mode: str = "auto") -> list[dict]:
    """
    Search the LinkedIn knowledge base.
    
    Args:
        query: Natural language search query
        mode: Search mode - "auto", "semantic", "structured", "graph"
    
    Returns:
        List of matching profiles with relevance info
    """
    orchestrator = get_orchestrator()
    result = orchestrator.search(query, mode=mode)
    return result.results

@mcp.tool()
def get_person_info(name: str) -> dict:
    """
    Get detailed info about a person.
    
    Args:
        name: Person's name
    
    Returns:
        Profile data including experience, skills, and graph connections
    """
    # Combine profile data with graph relationships
    ...

@mcp.tool()
def get_company_network(company: str) -> dict:
    """
    Get all people connected to a company.
    
    Args:
        company: Company name
    
    Returns:
        List of current/past employees, followers, etc.
    """
    ...

@mcp.tool()
def find_connection_path(person_a: str, person_b: str) -> list[str]:
    """
    Find relationship path between two people.
    
    Args:
        person_a: First person's name
        person_b: Second person's name
    
    Returns:
        List of people/companies forming the connection path
    """
    ...
```

---

## Execution Checklist
- status: todo
- type: task
- label: ['planning']
<!-- content -->

### Phase 1 Deliverables
- id: linkedin_kb.checklist.phase_1
- status: todo
- type: task
- label: ['planning']
<!-- content -->
- [ ] Project directory structure created.
- [ ] `requirements.txt` with all dependencies.
- [ ] `config/settings.yaml` with sensible defaults.
- [ ] `src/parser/schemas.py` with profile models.
- [ ] `src/graph/models.py` with node/edge models.
- [ ] Basic `README.md`.

### Phase 2 Deliverables
- id: linkedin_kb.checklist.phase_2
- status: todo
- type: task
- blocked_by: [linkedin_kb.checklist.phase_1]
- label: ['planning']
<!-- content -->
- [ ] `src/collector/session.py` — cookie management working.
- [ ] `src/collector/rate_limiter.py` — delays and daily cap working.
- [ ] `src/collector/fetcher.py` — can fetch a profile page.
- [ ] Manual test: successfully fetch own LinkedIn profile.

### Phase 3 Deliverables
- id: linkedin_kb.checklist.phase_3
- status: todo
- type: task
- blocked_by: [linkedin_kb.checklist.phase_2]
- label: ['planning']
<!-- content -->
- [ ] `src/parser/extractor.py` — extracts all profile sections.
- [ ] `src/parser/chunker.py` — creates embeddable chunks.
- [ ] `src/parser/relationship_extractor.py` — extracts edges.
- [ ] Manual test: parse own profile into structured data + edges.

### Phase 4 Deliverables
- id: linkedin_kb.checklist.phase_4
- status: todo
- type: task
- blocked_by: [linkedin_kb.checklist.phase_3]
- label: ['planning']
<!-- content -->
- [ ] `src/storage/raw_store.py` — JSON persistence working.
- [ ] `src/storage/vector_store.py` — ChromaDB integration working.
- [ ] `src/storage/structured_store.py` — DuckDB integration working.
- [ ] Manual test: embed and store 5 profiles.

### Phase 5 Deliverables
- id: linkedin_kb.checklist.phase_5
- status: todo
- type: task
- blocked_by: [linkedin_kb.checklist.phase_4]
- label: ['planning']
<!-- content -->
- [ ] `src/search/semantic.py` — similarity search working.
- [ ] `src/search/structured.py` — SQL queries working.
- [ ] `src/cli.py` — basic commands working.
- [ ] Manual test: search returns relevant results.

### Phase 6 Deliverables
- id: linkedin_kb.checklist.phase_6
- status: todo
- type: task
- blocked_by: [linkedin_kb.checklist.phase_5]
- label: ['planning']
<!-- content -->
- [ ] All unit tests passing.
- [ ] Documentation complete.
- [ ] Demo notebook functional.

### Phase 7 Deliverables
- id: linkedin_kb.checklist.phase_7
- status: todo
- type: task
- blocked_by: [linkedin_kb.checklist.phase_4]
- label: ['planning']
<!-- content -->
- [ ] `src/graph/builder.py` — profile-to-graph conversion working.
- [ ] `src/graph/writers.py` — MD + JSON export working.
- [ ] `src/graph/parser.py` — MD + JSON import working.
- [ ] `src/graph/utils.py` — traversal queries working.
- [ ] `data/graph/social_graph.md` — readable graph file.
- [ ] Manual test: query colleagues, company networks.

### Phase 8 Deliverables
- id: linkedin_kb.checklist.phase_8
- status: todo
- type: task
- blocked_by: [linkedin_kb.checklist.phase_7, linkedin_kb.checklist.phase_5]
- label: ['planning']
<!-- content -->
- [ ] `src/search/hybrid.py` — query routing working.
- [ ] `src/mcp/server.py` — MCP tools exposed.
- [ ] Manual test: LLM can query the knowledge base via MCP.

---

## Example Queries the System Should Handle
- status: active
- type: documentation
- label: ['planning']
<!-- content -->

### Semantic Queries (Vector Search)
- status: active
- type: documentation
- label: ['planning']
<!-- content -->
- "ML engineers with startup experience"
- "People interested in philosophy of AI"
- "Product managers who've worked in fintech"

### Structured Queries (DuckDB)
- status: active
- type: documentation
- label: ['planning']
<!-- content -->
- "Who currently works at Google?"
- "How many connections do I have at each company?"
- "List everyone with Python in their skills"

### Graph Queries (Relationship Traversal)
- status: active
- type: documentation
- label: ['planning']
<!-- content -->
- "Who are John's current colleagues?"
- "Find the connection path between me and Jane"
- "Who do I know at companies that Jane has worked at?"
- "Which of my connections also went to Stanford?"

### Hybrid Queries (Multiple Sources)
- status: active
- type: documentation
- label: ['planning']
<!-- content -->
- "ML engineers at Google who I might know through Stanford connections"
- "People at Meta who've posted about AI ethics"
- "Startup founders in my network who previously worked at FAANG"

---

## Risk Mitigation
- status: active
- type: documentation
- label: ['planning']
<!-- content -->

### LinkedIn Detection
- status: active
- type: documentation
- label: ['planning']
<!-- content -->
**Risk:** Account ban or soft block.
**Mitigations:**
1. Very slow rate limiting (45-120s between requests).
2. Daily cap (start with 20-30 profiles/day max).
3. Human-like patterns: occasional long pauses, session breaks.
4. Use your real account cookies (not automation-created sessions).
5. Don't scrape profiles you're not connected to (higher detection risk).

### HTML Changes
- status: active
- type: documentation
- label: ['planning']
<!-- content -->
**Risk:** LinkedIn updates HTML, breaking parser.
**Mitigations:**
1. Store raw HTML for re-parsing.
2. Use multiple fallback selectors.
3. Log extraction failures for quick diagnosis.
4. Design graceful degradation (partial data > crash).

### Legal Considerations
- status: active
- type: documentation
- label: ['planning']
<!-- content -->
**Risk:** Terms of Service violation.
**Mitigations:**
1. Personal use only, never commercial.
2. Don't redistribute collected data.
3. Only collect profiles you have legitimate interest in (your network).
4. Respect robots.txt spirit (even if not legally binding).

---

## Future Enhancements
- status: backlog
- type: documentation
- label: ['planning']
<!-- content -->
These are out of scope for initial implementation but worth noting:

1. **Activity tracking** — Track engagement (likes, comments) over time.
2. **Post/article collection** — Extend to content, not just profiles.
3. **Company page scraping** — Rich company data beyond what's on profiles.
4. **Auto-update** — Periodic refresh of stale profiles.
5. **Notification system** — Alert when a contact changes jobs.
6. **Visualization** — Interactive network graph (D3.js / Plotly).
7. **Export to your RAG system** — Direct integration with existing stack.
