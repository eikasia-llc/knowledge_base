# Unified Local Nexus: RAG + Data Warehouse Architecture
- status: active
- type: plan
- id: unified-nexus
- context_dependencies: {"conventions": "MD_CONVENTIONS.md", "rag_agent": "RAGS_AGENT.md"}
- last_checked: 2025-01-27
<!-- content -->
This document proposes an architecture that unifies **RAG (Retrieval-Augmented Generation)** for unstructured data with **Data Warehouse + Text2SQL** for structured data. The result is a system that can answer questions requiring both semantic understanding and precise computation.

## The Core Insight
- status: active
- type: context
- id: unified-nexus.insight
<!-- content -->
RAG and Data Warehouse approaches are **complementary, not competing**:

| Capability | RAG (Unstructured) | Data Warehouse (Structured) |
|:-----------|:-------------------|:----------------------------|
| **Data Type** | Documents, text, PDFs | Tables, CSVs, databases |
| **Query Style** | Semantic similarity | Exact SQL computation |
| **Strengths** | Fuzzy matching, context | Aggregations, joins, filters |
| **Weaknesses** | Can't compute, count | Can't understand meaning |
| **Example** | "What's our PTO policy?" | "How many sales in Q3?" |

**The gap**: Many real questions require BOTH:
- "Which product has the most complaints about shipping?" (semantic + aggregation)
- "Compare our revenue trends with what the market report says" (structured + unstructured)
- "Find customers who mentioned pricing concerns and have > $10K lifetime value" (semantic filter + computation)

This is the **TAG (Table-Augmented Generation)** paradigm emerging from Berkeley/Databricks research.

## Proposed Architecture
- status: active
- type: plan
- id: unified-nexus.architecture
<!-- content -->
The unified system has three retrieval paths that feed into a single LLM for answer generation:

```
                          ┌─────────────────────────────────────┐
                          │           User Question             │
                          └──────────────┬──────────────────────┘
                                         │
                                         ▼
                          ┌─────────────────────────────────────┐
                          │         Query Router (LLM)          │
                          │  Classifies: structured/unstructured│
                          │              /hybrid                │
                          └──────────────┬──────────────────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              │                          │                          │
              ▼                          ▼                          ▼
    ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
    │   Vector Store  │      │    DuckDB       │      │   Hybrid Path   │
    │   (ChromaDB)    │      │  (Text2SQL)     │      │  (Both + Join)  │
    │                 │      │                 │      │                 │
    │  Unstructured   │      │   Structured    │      │   Combined      │
    │  Documents      │      │   Tables        │      │   Reasoning     │
    └────────┬────────┘      └────────┬────────┘      └────────┬────────┘
             │                        │                        │
             └────────────────────────┼────────────────────────┘
                                      │
                                      ▼
                          ┌─────────────────────────────────────┐
                          │     Context Assembly & Generation   │
                          │  (LLM synthesizes all retrieved     │
                          │   information into final answer)    │
                          └─────────────────────────────────────┘
```

### Component Summary
- status: active
- type: context
- id: unified-nexus.architecture.components
<!-- content -->

| Component | Technology | Purpose |
|:----------|:-----------|:--------|
| **Document Store** | ChromaDB | Vector embeddings for semantic search |
| **Data Warehouse** | DuckDB | SQL queries over structured data |
| **Query Router** | LLM (cheap model) | Classify query type, route appropriately |
| **Text2SQL Engine** | LLM + schema context | Generate SQL from natural language |
| **Hybrid Executor** | Custom orchestrator | Combine SQL results with semantic search |
| **Answer Generator** | LLM (quality model) | Synthesize final response |

## Implementation Plan
- status: active
- type: plan
- id: unified-nexus.implementation
<!-- content -->
Starting from the existing `local_nexus` codebase (DuckDB + Streamlit), add RAG capabilities incrementally.

### Phase 1: Add Vector Store
- status: todo
- type: task
- id: unified-nexus.implementation.phase1
- priority: high
- estimate: 2h
<!-- content -->
Add ChromaDB alongside DuckDB to store document embeddings.

**File: `src/core/vector_store.py`**

```python
"""
Vector store for unstructured document storage and retrieval.

This module adds RAG capabilities to the Local Nexus data warehouse,
enabling semantic search over documents alongside SQL queries.
"""

import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from typing import Optional
import hashlib


class VectorStore:
    """
    Manages vector embeddings for unstructured documents.
    
    Uses ChromaDB with sentence-transformers for local embedding generation.
    Supports document ingestion, deduplication, and semantic search.
    """
    
    def __init__(self, persist_directory: str = "data/vectordb"):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Path to persist ChromaDB data
        """
        self.persist_dir = Path(persist_directory)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        
        # Use a local embedding model (no API costs)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create or get the main collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
    
    def _hash_content(self, content: str) -> str:
        """Generate SHA-256 hash for content deduplication."""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def add_document(
        self, 
        content: str, 
        metadata: Optional[dict] = None,
        doc_id: Optional[str] = None
    ) -> str:
        """
        Add a document to the vector store.
        
        Args:
            content: Document text content
            metadata: Optional metadata dict (source, date, type, etc.)
            doc_id: Optional explicit ID (generated from hash if not provided)
        
        Returns:
            The document ID (either provided or generated)
        """
        # Generate ID from content hash if not provided
        if doc_id is None:
            doc_id = self._hash_content(content)
        
        # Check for duplicates
        existing = self.collection.get(ids=[doc_id])
        if existing['ids']:
            return doc_id  # Already exists, skip
        
        # Add to collection
        self.collection.add(
            documents=[content],
            metadatas=[metadata or {}],
            ids=[doc_id]
        )
        
        return doc_id
    
    def add_chunks(
        self, 
        chunks: list[str], 
        metadatas: Optional[list[dict]] = None,
        source_id: Optional[str] = None
    ) -> list[str]:
        """
        Add multiple document chunks to the vector store.
        
        Args:
            chunks: List of text chunks
            metadatas: Optional list of metadata dicts (one per chunk)
            source_id: Optional source document identifier
        
        Returns:
            List of chunk IDs
        """
        ids = []
        metas = metadatas or [{}] * len(chunks)
        
        for i, (chunk, meta) in enumerate(zip(chunks, metas)):
            # Add source tracking to metadata
            chunk_meta = {**meta}
            if source_id:
                chunk_meta['source_id'] = source_id
                chunk_meta['chunk_index'] = i
            
            chunk_id = self.add_document(chunk, chunk_meta)
            ids.append(chunk_id)
        
        return ids
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        where: Optional[dict] = None
    ) -> list[dict]:
        """
        Semantic search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            where: Optional ChromaDB where filter
        
        Returns:
            List of dicts with 'id', 'text', 'metadata', 'distance'
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where
        )
        
        # Flatten results into list of dicts
        documents = []
        for i in range(len(results['ids'][0])):
            documents.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                'distance': results['distances'][0][i] if results['distances'] else None
            })
        
        return documents
    
    def get_stats(self) -> dict:
        """Get collection statistics."""
        return {
            'total_documents': self.collection.count(),
            'persist_directory': str(self.persist_dir)
        }
```

### Phase 2: Query Router
- status: todo
- type: task
- id: unified-nexus.implementation.phase2
- priority: high
- estimate: 1h
- blocked_by: [unified-nexus.implementation.phase1]
<!-- content -->
Classify incoming queries to route them to the appropriate retrieval path.

**File: `src/core/query_router.py`**

```python
"""
Query router for the unified RAG + Data Warehouse system.

Classifies queries and routes them to the appropriate retrieval path:
- Structured: SQL queries over DuckDB tables
- Unstructured: Semantic search over ChromaDB documents
- Hybrid: Both paths combined
"""

from enum import Enum
from typing import Optional
import re


class QueryType(Enum):
    """Classification of query types."""
    STRUCTURED = "structured"      # Pure SQL/computation
    UNSTRUCTURED = "unstructured"  # Pure semantic/document search
    HYBRID = "hybrid"              # Requires both


class QueryRouter:
    """
    Routes queries to the appropriate retrieval system.
    
    Uses a combination of heuristics and LLM classification
    to determine the optimal retrieval path.
    """
    
    # Keywords that strongly suggest structured queries
    STRUCTURED_KEYWORDS = {
        'how many', 'count', 'total', 'sum', 'average', 'avg',
        'maximum', 'minimum', 'max', 'min', 'top', 'bottom',
        'percentage', 'percent', 'ratio', 'compare numbers',
        'sales', 'revenue', 'cost', 'profit', 'quantity',
        'between dates', 'last month', 'this year', 'q1', 'q2', 'q3', 'q4',
        'group by', 'per', 'each', 'breakdown'
    }
    
    # Keywords that strongly suggest unstructured queries
    UNSTRUCTURED_KEYWORDS = {
        'what is', 'explain', 'describe', 'why', 'how does',
        'policy', 'procedure', 'guideline', 'documentation',
        'meaning', 'definition', 'concept', 'overview',
        'tell me about', 'information on', 'details about'
    }
    
    # Keywords suggesting hybrid queries
    HYBRID_KEYWORDS = {
        'which', 'who', 'find', 'identify', 'list',
        'mentioned', 'related to', 'about', 'with',
        'complaints', 'feedback', 'reviews', 'comments',
        'and also', 'as well as', 'combined with'
    }
    
    def __init__(self, llm_client=None, use_llm: bool = True):
        """
        Initialize the query router.
        
        Args:
            llm_client: Optional LLM client for classification
            use_llm: Whether to use LLM for ambiguous cases
        """
        self.llm_client = llm_client
        self.use_llm = use_llm and llm_client is not None
    
    def _heuristic_classify(self, query: str) -> tuple[QueryType, float]:
        """
        Classify query using keyword heuristics.
        
        Args:
            query: User query string
        
        Returns:
            Tuple of (QueryType, confidence score 0-1)
        """
        query_lower = query.lower()
        
        # Count keyword matches
        structured_score = sum(
            1 for kw in self.STRUCTURED_KEYWORDS 
            if kw in query_lower
        )
        unstructured_score = sum(
            1 for kw in self.UNSTRUCTURED_KEYWORDS 
            if kw in query_lower
        )
        hybrid_score = sum(
            1 for kw in self.HYBRID_KEYWORDS 
            if kw in query_lower
        )
        
        total = structured_score + unstructured_score + hybrid_score + 0.1
        
        # Determine type based on scores
        if hybrid_score > 0 and (structured_score > 0 or unstructured_score > 0):
            return QueryType.HYBRID, hybrid_score / total
        elif structured_score > unstructured_score:
            return QueryType.STRUCTURED, structured_score / total
        elif unstructured_score > structured_score:
            return QueryType.UNSTRUCTURED, unstructured_score / total
        else:
            # Ambiguous - default to hybrid for safety
            return QueryType.HYBRID, 0.3
    
    def _llm_classify(self, query: str) -> QueryType:
        """
        Use LLM to classify ambiguous queries.
        
        Args:
            query: User query string
        
        Returns:
            QueryType classification
        """
        if not self.llm_client:
            return QueryType.HYBRID
        
        prompt = f"""Classify this query into one of three categories:

Query: "{query}"

Categories:
- STRUCTURED: Requires numerical computation, aggregation, counting, or SQL-like operations over tabular data
- UNSTRUCTURED: Requires finding information in documents, policies, or text content
- HYBRID: Requires both computation on data AND understanding of text/documents

Respond with ONLY one word: STRUCTURED, UNSTRUCTURED, or HYBRID"""
        
        response = self.llm_client.generate_content(prompt)
        result = response.text.strip().upper()
        
        if 'STRUCTURED' in result:
            return QueryType.STRUCTURED
        elif 'UNSTRUCTURED' in result:
            return QueryType.UNSTRUCTURED
        else:
            return QueryType.HYBRID
    
    def classify(self, query: str) -> QueryType:
        """
        Classify a query to determine the retrieval path.
        
        Args:
            query: User query string
        
        Returns:
            QueryType indicating the recommended retrieval path
        """
        # First, try heuristic classification
        query_type, confidence = self._heuristic_classify(query)
        
        # If confident enough, use heuristic result
        if confidence > 0.5:
            return query_type
        
        # Otherwise, use LLM if available
        if self.use_llm:
            return self._llm_classify(query)
        
        # Default to hybrid for safety
        return query_type
    
    def get_classification_details(self, query: str) -> dict:
        """
        Get detailed classification information for debugging.
        
        Args:
            query: User query string
        
        Returns:
            Dict with classification details
        """
        query_type, confidence = self._heuristic_classify(query)
        
        return {
            'query': query,
            'heuristic_type': query_type.value,
            'heuristic_confidence': confidence,
            'final_type': self.classify(query).value,
            'used_llm': confidence <= 0.5 and self.use_llm
        }
```

### Phase 3: Text2SQL Engine
- status: todo
- type: task
- id: unified-nexus.implementation.phase3
- priority: high
- estimate: 2h
- blocked_by: [unified-nexus.implementation.phase2]
<!-- content -->
Generate SQL queries from natural language using schema context.

**File: `src/core/text2sql.py`**

```python
"""
Text-to-SQL engine for the Local Nexus data warehouse.

Converts natural language queries into SQL that can be executed
against DuckDB. Uses schema information and example queries
for accurate generation.
"""

import duckdb
from typing import Optional
import json


class Text2SQLEngine:
    """
    Converts natural language to SQL queries.
    
    Uses the database schema and optional example queries
    to generate accurate SQL. Includes validation and
    error handling for generated queries.
    """
    
    def __init__(self, db_path: str, llm_client):
        """
        Initialize the Text2SQL engine.
        
        Args:
            db_path: Path to DuckDB database file
            llm_client: LLM client for SQL generation
        """
        self.db_path = db_path
        self.llm = llm_client
        self._schema_cache = None
    
    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get a DuckDB connection."""
        return duckdb.connect(self.db_path, read_only=True)
    
    def get_schema(self, force_refresh: bool = False) -> dict:
        """
        Get the database schema for all tables.
        
        Args:
            force_refresh: Force refresh the cached schema
        
        Returns:
            Dict mapping table names to column info
        """
        if self._schema_cache and not force_refresh:
            return self._schema_cache
        
        schema = {}
        
        with self._get_connection() as conn:
            # Get all tables
            tables = conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
            
            for (table_name,) in tables:
                # Get column info for each table
                columns = conn.execute(f"""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = '{table_name}'
                """).fetchall()
                
                schema[table_name] = [
                    {
                        'name': col[0],
                        'type': col[1],
                        'nullable': col[2] == 'YES'
                    }
                    for col in columns
                ]
                
                # Get sample values for better context
                try:
                    sample = conn.execute(
                        f"SELECT * FROM {table_name} LIMIT 3"
                    ).fetchdf()
                    schema[table_name + '_sample'] = sample.to_dict('records')
                except Exception:
                    pass
        
        self._schema_cache = schema
        return schema
    
    def _format_schema_for_prompt(self) -> str:
        """Format schema information for the LLM prompt."""
        schema = self.get_schema()
        
        lines = ["Database Schema:"]
        for table_name, columns in schema.items():
            if table_name.endswith('_sample'):
                continue
            
            lines.append(f"\nTable: {table_name}")
            lines.append("Columns:")
            for col in columns:
                nullable = " (nullable)" if col['nullable'] else ""
                lines.append(f"  - {col['name']}: {col['type']}{nullable}")
            
            # Add sample if available
            sample_key = table_name + '_sample'
            if sample_key in schema and schema[sample_key]:
                lines.append(f"Sample data: {json.dumps(schema[sample_key][:2], default=str)}")
        
        return "\n".join(lines)
    
    def generate_sql(self, question: str) -> tuple[str, str]:
        """
        Generate SQL from a natural language question.
        
        Args:
            question: Natural language question
        
        Returns:
            Tuple of (sql_query, explanation)
        """
        schema_text = self._format_schema_for_prompt()
        
        prompt = f"""{schema_text}

User Question: "{question}"

Generate a DuckDB SQL query to answer this question.

Rules:
1. Use only tables and columns that exist in the schema
2. Use DuckDB syntax (similar to PostgreSQL)
3. Include appropriate JOINs if multiple tables are needed
4. Use meaningful column aliases for readability
5. Limit results to 100 rows unless aggregating

Respond in this exact format:
SQL:
```sql
YOUR_QUERY_HERE
```
EXPLANATION: Brief explanation of what the query does"""
        
        response = self.llm.generate_content(prompt)
        text = response.text
        
        # Parse SQL from response
        sql = ""
        explanation = ""
        
        if "```sql" in text:
            start = text.find("```sql") + 6
            end = text.find("```", start)
            sql = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            sql = text[start:end].strip()
        
        if "EXPLANATION:" in text:
            explanation = text.split("EXPLANATION:")[-1].strip()
        
        return sql, explanation
    
    def execute_sql(self, sql: str) -> tuple[list[dict], Optional[str]]:
        """
        Execute a SQL query and return results.
        
        Args:
            sql: SQL query to execute
        
        Returns:
            Tuple of (results as list of dicts, error message if any)
        """
        try:
            with self._get_connection() as conn:
                df = conn.execute(sql).fetchdf()
                return df.to_dict('records'), None
        except Exception as e:
            return [], str(e)
    
    def query(self, question: str) -> dict:
        """
        Full pipeline: question -> SQL -> results.
        
        Args:
            question: Natural language question
        
        Returns:
            Dict with sql, explanation, results, and any errors
        """
        # Generate SQL
        sql, explanation = self.generate_sql(question)
        
        if not sql:
            return {
                'success': False,
                'error': 'Could not generate SQL from question',
                'sql': None,
                'explanation': None,
                'results': []
            }
        
        # Execute SQL
        results, error = self.execute_sql(sql)
        
        return {
            'success': error is None,
            'sql': sql,
            'explanation': explanation,
            'results': results,
            'error': error
        }
```

### Phase 4: Unified Engine
- status: todo
- type: task
- id: unified-nexus.implementation.phase4
- priority: high
- estimate: 3h
- blocked_by: [unified-nexus.implementation.phase3]
<!-- content -->
Combine all components into a unified query engine.

**File: `src/core/unified_engine.py`**

```python
"""
Unified RAG + Data Warehouse engine.

This is the main orchestrator that combines:
- Query routing (classify query type)
- Vector store (semantic search for documents)
- Text2SQL (SQL queries for structured data)
- Answer generation (synthesize final response)
"""

from typing import Optional
from .vector_store import VectorStore
from .text2sql import Text2SQLEngine
from .query_router import QueryRouter, QueryType


class UnifiedEngine:
    """
    Unified query engine combining RAG and Data Warehouse capabilities.
    
    Routes queries to the appropriate retrieval path(s) and
    synthesizes a final answer from the retrieved context.
    """
    
    def __init__(
        self,
        db_path: str,
        vector_store_path: str,
        llm_client,
        cheap_llm_client=None
    ):
        """
        Initialize the unified engine.
        
        Args:
            db_path: Path to DuckDB database
            vector_store_path: Path to ChromaDB persistence
            llm_client: LLM client for answer generation
            cheap_llm_client: Optional cheaper LLM for routing/SQL
        """
        self.llm = llm_client
        self.cheap_llm = cheap_llm_client or llm_client
        
        # Initialize components
        self.vector_store = VectorStore(vector_store_path)
        self.text2sql = Text2SQLEngine(db_path, self.cheap_llm)
        self.router = QueryRouter(self.cheap_llm)
    
    def _retrieve_structured(self, question: str) -> dict:
        """
        Retrieve context via Text2SQL.
        
        Args:
            question: User question
        
        Returns:
            Dict with SQL results and metadata
        """
        result = self.text2sql.query(question)
        
        return {
            'type': 'structured',
            'success': result['success'],
            'sql': result.get('sql'),
            'data': result.get('results', []),
            'explanation': result.get('explanation'),
            'error': result.get('error')
        }
    
    def _retrieve_unstructured(self, question: str, top_k: int = 5) -> dict:
        """
        Retrieve context via semantic search.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
        
        Returns:
            Dict with retrieved documents
        """
        documents = self.vector_store.search(question, top_k=top_k)
        
        return {
            'type': 'unstructured',
            'success': len(documents) > 0,
            'documents': documents,
            'error': None if documents else 'No relevant documents found'
        }
    
    def _retrieve_hybrid(self, question: str) -> dict:
        """
        Retrieve context from both structured and unstructured sources.
        
        Args:
            question: User question
        
        Returns:
            Dict with combined results
        """
        structured = self._retrieve_structured(question)
        unstructured = self._retrieve_unstructured(question)
        
        return {
            'type': 'hybrid',
            'structured': structured,
            'unstructured': unstructured,
            'success': structured['success'] or unstructured['success']
        }
    
    def _format_context(self, retrieval_result: dict) -> str:
        """
        Format retrieved context for the answer generation prompt.
        
        Args:
            retrieval_result: Result from one of the retrieve methods
        
        Returns:
            Formatted context string
        """
        lines = []
        
        if retrieval_result['type'] == 'structured':
            if retrieval_result['success']:
                lines.append("=== DATABASE QUERY RESULTS ===")
                lines.append(f"SQL: {retrieval_result['sql']}")
                lines.append(f"Explanation: {retrieval_result['explanation']}")
                lines.append(f"Data ({len(retrieval_result['data'])} rows):")
                
                # Format data as readable table
                for i, row in enumerate(retrieval_result['data'][:20]):
                    lines.append(f"  {i+1}. {row}")
                
                if len(retrieval_result['data']) > 20:
                    lines.append(f"  ... and {len(retrieval_result['data']) - 20} more rows")
            else:
                lines.append(f"Database query failed: {retrieval_result['error']}")
        
        elif retrieval_result['type'] == 'unstructured':
            if retrieval_result['success']:
                lines.append("=== DOCUMENT SEARCH RESULTS ===")
                for i, doc in enumerate(retrieval_result['documents']):
                    lines.append(f"\n--- Document {i+1} ---")
                    if doc.get('metadata'):
                        lines.append(f"Source: {doc['metadata']}")
                    lines.append(doc['text'])
            else:
                lines.append("No relevant documents found.")
        
        elif retrieval_result['type'] == 'hybrid':
            # Format both structured and unstructured
            lines.append(self._format_context(retrieval_result['structured']))
            lines.append("\n")
            lines.append(self._format_context(retrieval_result['unstructured']))
        
        return "\n".join(lines)
    
    def _generate_answer(self, question: str, context: str, query_type: QueryType) -> str:
        """
        Generate final answer from retrieved context.
        
        Args:
            question: Original user question
            context: Formatted retrieval context
            query_type: Type of query (for tailored instructions)
        
        Returns:
            Generated answer string
        """
        type_instructions = {
            QueryType.STRUCTURED: "Focus on the numerical data and computations. Be precise with numbers.",
            QueryType.UNSTRUCTURED: "Synthesize information from the documents. Quote relevant passages.",
            QueryType.HYBRID: "Combine insights from both the data and documents. Connect numerical findings with contextual information."
        }
        
        prompt = f"""Answer the user's question based on the retrieved information.

Question: {question}

Retrieved Context:
{context}

Instructions:
- {type_instructions.get(query_type, '')}
- If the context doesn't contain enough information, say so clearly
- Cite specific data points or document sources when relevant
- Be concise but thorough

Your answer:"""
        
        response = self.llm.generate_content(prompt)
        return response.text
    
    def query(self, question: str) -> dict:
        """
        Process a user question through the unified system.
        
        This is the main entry point. It:
        1. Classifies the query type
        2. Routes to appropriate retrieval path(s)
        3. Generates a synthesized answer
        
        Args:
            question: User's natural language question
        
        Returns:
            Dict with answer and metadata
        """
        # Step 1: Classify the query
        query_type = self.router.classify(question)
        
        # Step 2: Retrieve context based on type
        if query_type == QueryType.STRUCTURED:
            retrieval = self._retrieve_structured(question)
        elif query_type == QueryType.UNSTRUCTURED:
            retrieval = self._retrieve_unstructured(question)
        else:  # HYBRID
            retrieval = self._retrieve_hybrid(question)
        
        # Step 3: Format context
        context = self._format_context(retrieval)
        
        # Step 4: Generate answer
        answer = self._generate_answer(question, context, query_type)
        
        return {
            'question': question,
            'query_type': query_type.value,
            'retrieval': retrieval,
            'context_summary': context[:500] + '...' if len(context) > 500 else context,
            'answer': answer
        }
    
    def get_system_stats(self) -> dict:
        """Get statistics about the unified system."""
        return {
            'vector_store': self.vector_store.get_stats(),
            'database_schema': list(self.text2sql.get_schema().keys())
        }
```

### Phase 5: Document Ingestion Pipeline
- status: todo
- type: task
- id: unified-nexus.implementation.phase5
- priority: medium
- estimate: 2h
- blocked_by: [unified-nexus.implementation.phase1]
<!-- content -->
Add document ingestion to complement the existing CSV/Excel ingestion.

**File: `src/core/document_ingestion.py`**

```python
"""
Document ingestion pipeline for the unified Local Nexus.

Handles ingestion of unstructured documents (PDF, TXT, DOCX, MD)
into the vector store, with chunking and metadata extraction.
"""

from pathlib import Path
from typing import Optional
import hashlib
import re


class DocumentChunker:
    """
    Chunks documents into smaller pieces for embedding.
    
    Supports multiple chunking strategies based on document type.
    """
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target chunk size in tokens (approximate)
            overlap: Number of tokens to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token)."""
        return len(text) // 4
    
    def chunk_by_size(self, text: str) -> list[str]:
        """
        Chunk text by fixed size with overlap.
        
        Args:
            text: Input text
        
        Returns:
            List of text chunks
        """
        # Convert target tokens to chars
        char_size = self.chunk_size * 4
        char_overlap = self.overlap * 4
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + char_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end within last 20% of chunk
                search_start = end - int(char_size * 0.2)
                search_region = text[search_start:end]
                
                # Find last sentence boundary
                for pattern in ['. ', '.\n', '? ', '! ']:
                    last_boundary = search_region.rfind(pattern)
                    if last_boundary != -1:
                        end = search_start + last_boundary + len(pattern)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - char_overlap
        
        return chunks
    
    def chunk_by_headers(self, text: str) -> list[str]:
        """
        Chunk markdown/text by headers.
        
        Useful for structured documents where each section
        should be its own chunk.
        
        Args:
            text: Input text with headers
        
        Returns:
            List of text chunks (one per section)
        """
        # Split on markdown headers
        pattern = r'(^#{1,6}\s+.+$)'
        parts = re.split(pattern, text, flags=re.MULTILINE)
        
        chunks = []
        current_chunk = ""
        
        for part in parts:
            if re.match(r'^#{1,6}\s+', part):
                # This is a header - start new chunk
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = part + "\n"
            else:
                current_chunk += part
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # If chunks are too big, split them further
        final_chunks = []
        for chunk in chunks:
            if self._estimate_tokens(chunk) > self.chunk_size * 1.5:
                final_chunks.extend(self.chunk_by_size(chunk))
            else:
                final_chunks.append(chunk)
        
        return final_chunks


class DocumentIngester:
    """
    Ingests documents into the vector store.
    
    Handles file reading, metadata extraction, chunking,
    and deduplication.
    """
    
    def __init__(self, vector_store, chunker: Optional[DocumentChunker] = None):
        """
        Initialize the ingester.
        
        Args:
            vector_store: VectorStore instance
            chunker: Optional custom chunker
        """
        self.vector_store = vector_store
        self.chunker = chunker or DocumentChunker()
    
    def _read_file(self, file_path: Path) -> tuple[str, str]:
        """
        Read file content based on extension.
        
        Args:
            file_path: Path to file
        
        Returns:
            Tuple of (content, file_type)
        """
        suffix = file_path.suffix.lower()
        
        if suffix == '.txt' or suffix == '.md':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read(), suffix[1:]
        
        elif suffix == '.pdf':
            try:
                import pypdf
                reader = pypdf.PdfReader(str(file_path))
                text = "\n".join(page.extract_text() for page in reader.pages)
                return text, 'pdf'
            except ImportError:
                raise ImportError("pypdf required for PDF ingestion: pip install pypdf")
        
        elif suffix == '.docx':
            try:
                from docx import Document
                doc = Document(str(file_path))
                text = "\n".join(para.text for para in doc.paragraphs)
                return text, 'docx'
            except ImportError:
                raise ImportError("python-docx required for DOCX ingestion: pip install python-docx")
        
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    
    def _generate_doc_id(self, content: str) -> str:
        """Generate unique document ID from content hash."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def ingest_file(self, file_path: str | Path) -> dict:
        """
        Ingest a single file into the vector store.
        
        Args:
            file_path: Path to the file
        
        Returns:
            Dict with ingestion results
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {'success': False, 'error': f'File not found: {file_path}'}
        
        # Read file
        try:
            content, file_type = self._read_file(file_path)
        except Exception as e:
            return {'success': False, 'error': str(e)}
        
        # Generate document ID
        doc_id = self._generate_doc_id(content)
        
        # Choose chunking strategy
        if file_type == 'md':
            chunks = self.chunker.chunk_by_headers(content)
        else:
            chunks = self.chunker.chunk_by_size(content)
        
        # Prepare metadata
        base_metadata = {
            'source_file': file_path.name,
            'file_type': file_type,
            'source_path': str(file_path),
            'total_chunks': len(chunks)
        }
        
        # Ingest chunks
        chunk_ids = self.vector_store.add_chunks(
            chunks=chunks,
            metadatas=[{**base_metadata, 'chunk_index': i} for i in range(len(chunks))],
            source_id=doc_id
        )
        
        return {
            'success': True,
            'document_id': doc_id,
            'file_name': file_path.name,
            'file_type': file_type,
            'chunks_created': len(chunk_ids),
            'chunk_ids': chunk_ids
        }
    
    def ingest_directory(self, directory: str | Path, extensions: list[str] = None) -> list[dict]:
        """
        Ingest all matching files in a directory.
        
        Args:
            directory: Path to directory
            extensions: List of extensions to process (default: all supported)
        
        Returns:
            List of ingestion results
        """
        directory = Path(directory)
        extensions = extensions or ['.txt', '.md', '.pdf', '.docx']
        
        results = []
        for ext in extensions:
            for file_path in directory.glob(f'**/*{ext}'):
                result = self.ingest_file(file_path)
                results.append(result)
        
        return results
```

## Usage Examples
- status: active
- type: context
- id: unified-nexus.examples
<!-- content -->
Examples of using the unified system.

### Basic Usage

```python
from src.core.unified_engine import UnifiedEngine
import google.generativeai as genai

# Configure LLM
genai.configure(api_key="your-api-key")
llm = genai.GenerativeModel('gemini-1.5-flash')

# Initialize unified engine
engine = UnifiedEngine(
    db_path="data/warehouse.db",
    vector_store_path="data/vectordb",
    llm_client=llm
)

# Query examples
# Structured query (routes to Text2SQL)
result = engine.query("What were our total sales last month?")
print(result['answer'])

# Unstructured query (routes to RAG)
result = engine.query("What is our refund policy?")
print(result['answer'])

# Hybrid query (uses both)
result = engine.query("Which customers complained about shipping and have orders over $500?")
print(result['answer'])
```

### Document Ingestion

```python
from src.core.document_ingestion import DocumentIngester
from src.core.vector_store import VectorStore

# Initialize
vector_store = VectorStore("data/vectordb")
ingester = DocumentIngester(vector_store)

# Ingest a single file
result = ingester.ingest_file("docs/company_policies.pdf")
print(f"Created {result['chunks_created']} chunks")

# Ingest all documents in a directory
results = ingester.ingest_directory("docs/", extensions=['.pdf', '.md'])
total_chunks = sum(r['chunks_created'] for r in results if r['success'])
print(f"Ingested {len(results)} files, {total_chunks} total chunks")
```

## Integration with Streamlit UI
- status: todo
- type: task
- id: unified-nexus.ui-integration
- priority: medium
- estimate: 2h
- blocked_by: [unified-nexus.implementation.phase4]
<!-- content -->
Update the Streamlit app to support both data types.

**Add to `src/app.py`:**

```python
import streamlit as st
from src.core.unified_engine import UnifiedEngine
from src.core.document_ingestion import DocumentIngester

# Sidebar - Data Sources section
with st.sidebar:
    st.header("Data Sources")
    
    # Existing: Structured data upload
    st.subheader("📊 Structured Data")
    uploaded_csv = st.file_uploader(
        "Upload CSV/Excel", 
        type=['csv', 'xlsx', 'xls']
    )
    
    # New: Document upload
    st.subheader("📄 Documents")
    uploaded_docs = st.file_uploader(
        "Upload Documents",
        type=['pdf', 'txt', 'md', 'docx'],
        accept_multiple_files=True
    )
    
    if uploaded_docs:
        for doc in uploaded_docs:
            # Save and ingest
            with open(f"data/uploads/{doc.name}", "wb") as f:
                f.write(doc.getbuffer())
            result = ingester.ingest_file(f"data/uploads/{doc.name}")
            if result['success']:
                st.success(f"✓ {doc.name}: {result['chunks_created']} chunks")
            else:
                st.error(f"✗ {doc.name}: {result['error']}")
    
    # Show system stats
    st.divider()
    st.subheader("System Status")
    stats = engine.get_system_stats()
    st.metric("Documents", stats['vector_store']['total_documents'])
    st.metric("Tables", len(stats['database_schema']))

# Chat interface
if prompt := st.chat_input("Ask about your data or documents..."):
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = engine.query(prompt)
        
        # Show query type badge
        type_colors = {
            'structured': '🔢',
            'unstructured': '📄',
            'hybrid': '🔀'
        }
        st.caption(f"{type_colors.get(result['query_type'], '')} Query type: {result['query_type']}")
        
        # Show answer
        st.write(result['answer'])
        
        # Expandable details
        with st.expander("View retrieval details"):
            st.json(result['retrieval'])
```

## Key Benefits of Unified Architecture
- status: active
- type: context
- id: unified-nexus.benefits
<!-- content -->

| Benefit | Description |
|:--------|:------------|
| **Single Interface** | Users ask questions naturally, system figures out how to answer |
| **Complementary Strengths** | SQL precision + semantic understanding |
| **Cost Effective** | Cheap LLM for routing/SQL, quality LLM only for final answer |
| **Local First** | Both DuckDB and ChromaDB run locally, no cloud dependency |
| **Incremental Adoption** | Add documents gradually alongside structured data |
| **Debuggable** | Clear separation of concerns, each component testable |

## Future Enhancements
- status: active
- type: context
- id: unified-nexus.future
<!-- content -->

1. **Graph Layer**: Add a knowledge graph for entity relationships
2. **Semantic SQL**: Use embeddings to help with table/column matching
3. **Feedback Loop**: Log queries and outcomes for fine-tuning
4. **Multi-modal**: Support images and diagrams in documents
5. **MCP Server**: Expose the unified engine as MCP tools for other agents
