# CLAUDE.md - AI Assistant Guide for A-MEM

 

> **Last Updated:** December 1, 2025

> **Version:** 1.0

> **Purpose:** Comprehensive guide for AI assistants working with the A-MEM codebase

 

---

 

## 📋 Table of Contents

 

1. [Project Overview](#project-overview)

2. [Architecture & Design Principles](#architecture--design-principles)

3. [Directory Structure](#directory-structure)

4. [Key Components & Modules](#key-components--modules)

5. [Development Workflows](#development-workflows)

6. [Configuration & Environment](#configuration--environment)

7. [MCP Server Integration](#mcp-server-integration)

8. [Testing Strategy](#testing-strategy)

9. [Common Tasks & Patterns](#common-tasks--patterns)

10. [Best Practices & Conventions](#best-practices--conventions)

11. [Troubleshooting Guide](#troubleshooting-guide)

 

---

 

## 🎯 Project Overview

 

### What is A-MEM?

 

**A-MEM** (Agentic Memory) is an **MCP-integrated memory system** for LLM agents based on the **Zettelkasten principle**. It provides persistent, graph-based memory with semantic retrieval capabilities for AI assistants in IDE environments (Cursor, VSCode).

 

### Research Foundation

 

Based on the paper: ["A-Mem: Agentic Memory for LLM Agents"](https://arxiv.org/html/2502.12110v11)

Authors: Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, Yongfeng Zhang

 

### Key Differentiators

 

This implementation focuses on:

- **MCP Protocol Integration** for IDE environments

- **Explicit Graph-Based Memory Linking** with typed edges, reasoning, and weights

- **Dual Storage Architecture**: ChromaDB (vector similarity) + Graph Backend (NetworkX/RustworkX/FalkorDB)

- **Autonomous Maintenance**: Memory Enzymes for graph health

- **Research Integration**: Web research agent for JIT context optimization

 

---

 

## 🏗️ Architecture & Design Principles

 

### Core Architecture Patterns

 

```

┌─────────────────────────────────────────────────────────────┐

│                      MCP Server Layer                        │

│              (src/a_mem/main.py - stdio_server)             │

└───────────────────────┬─────────────────────────────────────┘

                        │

                        ▼

┌─────────────────────────────────────────────────────────────┐

│                  Memory Controller Layer                     │

│              (src/a_mem/core/logic.py)                      │

│  • Async I/O via run_in_executor                            │

│  • Background evolution tasks                                │

│  • Enzyme scheduler orchestration                           │

└───────────┬─────────────────────────┬───────────────────────┘

            │                         │

            ▼                         ▼

┌───────────────────────┐  ┌──────────────────────────────────┐

│   Storage Layer       │  │      LLM Service Layer           │

│  (storage/engine.py)  │  │     (utils/llm.py)               │

│                       │  │                                  │

│  • GraphStore         │  │  • Metadata extraction           │

│  • VectorStore        │  │  • Embedding generation          │

│  • Cross-platform     │  │  • Multi-provider support        │

│    file locking       │  │    (Ollama/OpenRouter)           │

└───────────────────────┘  └──────────────────────────────────┘

```

 

### Design Principles

 

1. **Async Non-Blocking I/O**

   - All blocking operations use `asyncio.run_in_executor`

   - Background tasks for evolution and maintenance

   - Parallel HTTP server for external tool access

 

2. **Dual Storage Architecture**

   - **Vector Store (ChromaDB)**: Semantic similarity search

   - **Graph Store**: Explicit typed relationships with reasoning

   - Enables hybrid retrieval: similarity + graph traversal

 

3. **Graph Backend Flexibility**

   - **NetworkX** (default): Cross-platform, no extra dependencies

   - **RustworkX** (performance): 3x-100x faster, Windows-compatible

   - **FalkorDB** (experimental): Persistent storage, Redis-based

 

4. **Type Safety with Pydantic**

   - All data models use Pydantic BaseModel

   - Automatic validation and serialization

   - Type hints throughout codebase

 

5. **Event-Driven Architecture**

   - JSONL event log for audit trail (`data/events.jsonl`)

   - All critical operations logged

   - Enables debugging and analytics

 

---

 

## 📁 Directory Structure

 

```

a-mem-mcp-server/

├── src/

│   └── a_mem/

│       ├── __init__.py

│       ├── main.py                    # MCP server entry point

│       ├── config.py                  # Configuration & environment

│       ├── models/

│       │   ├── __init__.py

│       │   └── note.py                # Data models (AtomicNote, NoteInput, etc.)

│       ├── core/

│       │   ├── __init__.py

│       │   └── logic.py               # MemoryController (core business logic)

│       ├── storage/

│       │   ├── __init__.py

│       │   ├── engine.py              # StorageManager, GraphStore, VectorStore

│       │   ├── rustworkx_store.py     # RustworkX graph backend

│       │   ├── falkordb_store.py      # FalkorDB graph backend (Linux/macOS)

│       │   ├── falkordb_store_windows.py  # FalkorDB Windows adapter

│       │   └── safe_graph_wrapper.py  # Edge case handling wrapper

│       └── utils/

│           ├── __init__.py

│           ├── llm.py                 # LLM service (metadata, embeddings)

│           ├── priority.py            # Priority scoring & event logging

│           ├── enzymes.py             # Memory maintenance (14+ operations)

│           ├── researcher.py          # Web research agent

│           ├── researcher_tools.py    # HTTP-based research tools

│           ├── validation.py          # MCP parameter validation

│           └── serializers.py         # Data serialization helpers

├── tests/                             # Test suite (24+ tests)

│   ├── test_a_mem.py                 # Core functionality tests

│   ├── test_code_structure.py        # Structure validation

│   ├── test_new_features.py          # Type classification, priority, events

│   ├── test_enzymes.py               # Memory enzymes tests

│   ├── test_scheduler.py             # Scheduler tests

│   ├── test_mcp_integration.py       # MCP server integration tests

│   └── test_researcher*.py           # Researcher agent tests

├── tools/                             # Standalone utilities

│   ├── amem_stats.py                 # CLI status tool (like git status)

│   ├── visualize_memory.py           # Web-based graph dashboard

│   ├── extract_graph.py              # Graph data extractor

│   └── a_mem_cli.py                  # Command-line interface

├── docs/                              # Extended documentation

│   ├── MEMORY_ENZYMES_DETAILED.md    # Enzyme documentation

│   ├── RESEARCHER_AGENT_DETAILED.md  # Research agent guide

│   ├── TEST_REPORT.md                # Test results

│   ├── MCP_SERVER_TEST_REPORT.md     # MCP integration tests

│   └── *.svg                         # Architecture diagrams

├── data/                              # Runtime data (auto-created)

│   ├── chroma/                       # ChromaDB vector store

│   ├── graph/

│   │   ├── knowledge_graph.json      # Graph snapshot (NetworkX)

│   │   ├── knowledge_graph.graphml   # Graph snapshot (RustworkX)

│   │   └── graph.lock                # Cross-platform file lock

│   └── events.jsonl                  # Event log (append-only)

├── .env.example                       # Configuration template

├── requirements.txt                   # Python dependencies

├── mcp_server.py                      # MCP server launcher

├── README.md                          # User documentation

├── MCP_SERVER_SETUP.md               # MCP setup guide

└── CLAUDE.md                          # This file (AI assistant guide)

```

 

### Key Files to Understand

 

| File | Purpose | Key Classes/Functions |

|------|---------|----------------------|

| `src/a_mem/main.py` | MCP server implementation | `list_tools()`, `call_tool()`, `main()` |

| `src/a_mem/core/logic.py` | Core business logic | `MemoryController`, `create_note()`, `retrieve()` |

| `src/a_mem/storage/engine.py` | Storage layer | `StorageManager`, `GraphStore`, `VectorStore` |

| `src/a_mem/utils/llm.py` | LLM integration | `LLMService`, `extract_metadata()`, `get_embedding()` |

| `src/a_mem/utils/enzymes.py` | Memory maintenance | `run_memory_enzymes()`, 14+ enzyme functions |

| `src/a_mem/models/note.py` | Data models | `AtomicNote`, `NoteInput`, `NoteRelation` |

| `src/a_mem/config.py` | Configuration | `Config` class, environment variables |

 

---

 

## 🔧 Key Components & Modules

 

### 1. MCP Server (`src/a_mem/main.py`)

 

**Purpose:** JSON-RPC server implementing Model Context Protocol (MCP)

 

**Key Functions:**

- `list_tools()` - Returns 15 available MCP tools

- `call_tool(name, arguments)` - Routes tool calls to controller

- `main()` - Server initialization, enzyme scheduler, HTTP server

 

**15 MCP Tools:**

1. `create_atomic_note` - Store new memory

2. `retrieve_memories` - Semantic search with priority

3. `get_memory_stats` - System statistics

4. `add_file` - Import file with chunking

5. `reset_memory` - Clear all data

6. `list_notes` - List all notes

7. `get_note` - Get single note by ID

8. `update_note` - Update note metadata

9. `delete_atomic_note` - Delete note

10. `list_relations` - List graph edges

11. `add_relation` - Manual edge creation

12. `remove_relation` - Delete edge

13. `get_graph` - Full graph snapshot

14. `run_memory_enzymes` - Manual maintenance

15. `research_and_store` - Web research + storage

 

**Communication:**

- Uses `stdio_server` for IDE integration

- Logs to stderr (not stdout) to avoid breaking JSON-RPC

- Optional HTTP server on port 42424 for external tools

 

### 2. Memory Controller (`src/a_mem/core/logic.py`)

 

**Purpose:** Orchestrates memory operations with async I/O

 

**Key Methods:**

 

```python

class MemoryController:

    async def create_note(input_data: NoteInput) -> str:

        """

        1. Extract metadata via LLM (or use pre-provided)

        2. Generate embedding (concat: content + summary + keywords + tags)

        3. Store in vector DB + graph

        4. Log event

        5. Background evolution task

        """

 

    async def retrieve(query: str) -> List[SearchResult]:

        """

        1. Search vector DB for similar notes

        2. Compute priority scores (type + age + usage + edges)

        3. Traverse graph for connected notes

        4. Sort by combined score (similarity × priority)

        5. Return with context

        """

 

    async def _evolve_memory(note, embedding):

        """

        Background task:

        1. Find similar notes (cosine similarity)

        2. Create typed edges with reasoning

        3. Check for existing notes to merge/update

        4. Log evolution events

        """

```

 

**Async Pattern:**

```python

# All blocking I/O uses run_in_executor

loop = asyncio.get_running_loop()

result = await loop.run_in_executor(None, blocking_function, args)

```

 

### 3. Storage Layer (`src/a_mem/storage/engine.py`)

 

**GraphStore:**

```python

class GraphStore:

    def __init__(self):

        self.graph = nx.DiGraph()  # Or RustworkX/FalkorDB

        self.load()

 

    def add_node(self, note: AtomicNote):

        """Store node with all metadata"""

 

    def add_edge(self, source, target, relation_type, reasoning, weight):

        """Create typed edge with reasoning"""

 

    def save_snapshot(self):

        """Atomic save with temp file + rename"""

```

 

**VectorStore:**

```python

class VectorStore:

    def __init__(self):

        self.client = chromadb.PersistentClient(path=settings.CHROMA_DIR)

        self.collection = self.client.get_or_create_collection("a_mem_notes")

 

    def add(self, note: AtomicNote, embedding: List[float]):

        """Store with metadata for filtering"""

 

    def search(self, query_embedding, max_results=10):

        """Cosine similarity search"""

```

 

**Cross-Platform File Locking:**

- Uses `fcntl` on Linux/macOS

- Falls back to `portalocker` on Windows

- Prevents concurrent write conflicts

 

### 4. LLM Service (`src/a_mem/utils/llm.py`)

 

**Multi-Provider Support:**

 

```python

class LLMService:

    def __init__(self):

        self.provider = settings.LLM_PROVIDER  # "ollama" or "openrouter"

 

    def extract_metadata(self, content: str) -> dict:

        """

        Extracts:

        - contextual_summary

        - keywords (max 7)

        - tags

        - type (rule/procedure/concept/tool/reference/integration)

        """

 

    def get_embedding(self, text: str) -> List[float]:

        """Generate embedding vector"""

 

    def refine_summary(self, content: str, old_summary: str) -> str:

        """Make similar summaries more specific"""

```

 

**Provider Configuration:**

- **Ollama** (local): HTTP requests to localhost:11434

- **OpenRouter** (cloud): API key-based, OpenAI-compatible

 

### 5. Memory Enzymes (`src/a_mem/utils/enzymes.py`)

 

**14+ Autonomous Maintenance Operations:**

 

```python

def run_memory_enzymes(graph, llm, prune_config, suggest_config, refine_config):

    """

    1. Link Pruner: Remove old/weak edges (age > 90 days, weight < 0.3)

    2. Zombie Node Remover: Delete empty nodes

    3. Duplicate Merger: Find and merge exact/semantic duplicates

    4. Edge Validator: Fix edges (add reasoning, standardize types)

    5. Self-Loop Remover: Remove self-referential edges

    6. Isolated Node Finder: Identify unconnected nodes

    7. Isolated Node Linker: Auto-link isolated nodes (similarity ≥ 0.70)

    8. Keyword Normalizer: Clean and limit keywords (max 7)

    9. Quality Score Calculator: Score notes by content/metadata/connections

    10. Note Validator: Validate and correct missing fields

    11. Low Quality Note Remover: Remove CAPTCHA/error/spam pages

    12. Summary Refiner: Make similar summaries more distinct

    13. Corrupted Node Repairer: Fix nodes with invalid data

    14. Relation Suggester: Find semantic connections (similarity ≥ 0.75)

    15. Summary Digester: Compress nodes with >8 children

    """

```

 

**Scheduler:**

- Runs automatically every hour (configurable)

- Auto-saves graph every 5 minutes

- Graceful error handling

 

### 6. Research Agent (`src/a_mem/utils/researcher.py`)

 

**Purpose:** JIT web research for low-confidence queries

 

**Workflow:**

```python

class ResearcherAgent:

    async def research(query: str, context: str) -> List[AtomicNote]:

        """

        1. Search web (Google Search API or DuckDuckGo)

        2. Extract top N URLs

        3. Fetch content:

           - Web pages: Jina Reader (local/cloud) or Readability

           - PDFs: Unstructured (library/API)

        4. Parse and clean content

        5. Extract metadata via LLM

        6. Create AtomicNote objects

        7. Return for storage

        """

```

 

**Hybrid Tool Strategy:**

- **Primary**: MCP tools (if available via callback)

- **Fallback**: HTTP-based tools (Google API, DuckDuckGo, Jina Reader)

 

**Configuration:**

```bash

RESEARCHER_ENABLED=true

RESEARCHER_CONFIDENCE_THRESHOLD=0.5  # Auto-trigger when score < 0.5

RESEARCHER_MAX_SOURCES=5

GOOGLE_SEARCH_ENABLED=true

JINA_READER_ENABLED=true

UNSTRUCTURED_ENABLED=true

```

 

### 7. Data Models (`src/a_mem/models/note.py`)

 

**Core Models:**

 

```python

class AtomicNote(BaseModel):

    id: str                          # UUID

    content: str                     # Original text

    contextual_summary: str          # LLM-generated summary

    keywords: List[str]              # Max 7 keywords

    tags: List[str]                  # Categorical tags

    created_at: datetime             # Timestamp

    type: Optional[str]              # rule/procedure/concept/tool/reference/integration

    metadata: Dict[str, Any]         # Experimental fields

 

class NoteInput(BaseModel):

    content: str

    source: Optional[str] = "user_input"

    # Pre-extracted metadata (optional, from ResearcherAgent)

    contextual_summary: Optional[str] = None

    keywords: Optional[List[str]] = None

    tags: Optional[List[str]] = None

    type: Optional[str] = None

    metadata: Optional[Dict[str, Any]] = None

 

class NoteRelation(BaseModel):

    source_id: str

    target_id: str

    relation_type: str               # relates_to/contradicts/supports/etc.

    reasoning: Optional[str]         # Why this relation exists

    weight: float = 1.0              # 0.0-1.0

    created_at: datetime

 

class SearchResult(BaseModel):

    note: AtomicNote

    score: float                     # Combined similarity × priority

    related_notes: List[AtomicNote]  # Graph-connected notes

```

 

---

 

## 🔄 Development Workflows

 

### Adding a New MCP Tool

 

1. **Define tool schema** in `main.py:list_tools()`:

```python

Tool(

    name="my_new_tool",

    description="Clear description for AI assistants",

    inputSchema={

        "type": "object",

        "properties": {

            "param1": {"type": "string", "description": "..."},

        },

        "required": ["param1"]

    }

)

```

 

2. **Add handler** in `main.py:call_tool()`:

```python

elif name == "my_new_tool":

    param1 = arguments.get("param1", "")

    result = await controller.my_method(param1)

    return [TextContent(type="text", text=json.dumps(result, indent=2))]

```

 

3. **Implement logic** in `core/logic.py`:

```python

async def my_method(self, param1: str):

    loop = asyncio.get_running_loop()

    # Offload blocking I/O

    result = await loop.run_in_executor(None, self._blocking_operation, param1)

    return result

```

 

4. **Add tests** in `tests/test_mcp_server.py`

 

### Modifying Storage Backend

 

**GraphStore modifications:**

- Edit `storage/engine.py` for NetworkX

- Edit `storage/rustworkx_store.py` for RustworkX

- Edit `storage/falkordb_store.py` for FalkorDB

 

**Key methods to update:**

```python

def add_node(self, note: AtomicNote)

def add_edge(self, source, target, relation_type, reasoning, weight)

def get_neighbors(self, node_id) -> List[Tuple[str, dict]]

def save_snapshot()

def load()

```

 

**VectorStore modifications:**

- Edit `storage/engine.py:VectorStore`

- ChromaDB API: `add()`, `query()`, `delete()`

 

### Adding a Memory Enzyme

 

1. **Create enzyme function** in `utils/enzymes.py`:

```python

def my_enzyme(graph: GraphStore, llm: LLMService, config: dict) -> dict:

    """

    Args:

        graph: GraphStore instance

        llm: LLMService instance

        config: Enzyme-specific configuration

 

    Returns:

        dict: Results with counts, lists, etc.

    """

    # Your logic here

    log_event("MY_ENZYME_RUN", {"count": 42})

    return {"count": 42}

```

 

2. **Add to enzyme runner** in `utils/enzymes.py:run_memory_enzymes()`:

```python

results["my_enzyme"] = my_enzyme(graph, llm, config)

```

 

3. **Add to MCP tool parameters** (optional) in `main.py`

 

4. **Add tests** in `tests/test_enzymes.py`

 

### Debugging with Event Logs

 

**View recent events:**

```bash

tail -n 50 data/events.jsonl | jq .

```

 

**Filter by event type:**

```bash

grep "NOTE_CREATED" data/events.jsonl | jq .

```

 

**Event types:**

- `NOTE_CREATED`, `RELATION_CREATED`, `MEMORY_EVOLVED`

- `LINKS_PRUNED`, `RELATION_PRUNED`, `NODE_PRUNED`

- `DUPLICATES_MERGED`, `SELF_LOOPS_REMOVED`

- `ISOLATED_NODES_FOUND`, `ISOLATED_NODES_LINKED`

- `KEYWORDS_NORMALIZED`, `QUALITY_SCORES_CALCULATED`

- `NOTES_VALIDATED`, `LOW_QUALITY_NOTES_REMOVED`

- `CORRUPTED_NODES_REPAIRED`, `RELATIONS_SUGGESTED`

- `ENZYME_SCHEDULER_RUN`, `RESEARCHER_MANUAL_RUN`

 

---

 

## ⚙️ Configuration & Environment

 

### Environment Variables (`.env` file)

 

**LLM Provider:**

```bash

LLM_PROVIDER=ollama              # "ollama" or "openrouter"

 

# Ollama (local)

OLLAMA_BASE_URL=http://localhost:11434

OLLAMA_LLM_MODEL=qwen3:4b

OLLAMA_EMBEDDING_MODEL=nomic-embed-text:latest

 

# OpenRouter (cloud)

OPENROUTER_API_KEY=your_key_here

OPENROUTER_LLM_MODEL=openai/gpt-4o-mini

OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small

```

 

**Graph Backend:**

```bash

GRAPH_BACKEND=networkx           # "networkx", "rustworkx", or "falkordb"

```

 

**Retrieval:**

```bash

MAX_NEIGHBORS=5                  # Max connected notes per result

MIN_SIMILARITY_SCORE=0.4         # Minimum cosine similarity

```

 

**Research Agent:**

```bash

RESEARCHER_ENABLED=true

RESEARCHER_CONFIDENCE_THRESHOLD=0.5

RESEARCHER_MAX_SOURCES=5

GOOGLE_SEARCH_ENABLED=true

GOOGLE_API_KEY=your_key

GOOGLE_SEARCH_ENGINE_ID=your_id

JINA_READER_ENABLED=true

UNSTRUCTURED_ENABLED=true

```

 

**HTTP Server (optional):**

```bash

TCP_SERVER_ENABLED=false         # Enable HTTP endpoint for tools

TCP_SERVER_HOST=127.0.0.1

TCP_SERVER_PORT=42424

```

 

### Configuration Hierarchy

 

1. **Base:** `.env` file (default values)

2. **Override:** MCP `env` block in `mcp.json`

3. **Priority:** MCP env > .env file

 

**Example MCP config with overrides:**

```json

{

  "mcpServers": {

    "a-mem": {

      "command": "python",

      "args": ["-m", "src.a_mem.main"],

      "cwd": "/path/to/a-mem-mcp-server",

      "env": {

        "LLM_PROVIDER": "ollama",

        "OLLAMA_LLM_MODEL": "llama3.2:3b",

        "RESEARCHER_ENABLED": "true"

      }

    }

  }

}

```

 

---

 

## 🔌 MCP Server Integration

 

### MCP Protocol Basics

 

**Communication:**

- Uses **stdio** (stdin/stdout) for JSON-RPC messages

- All logging must go to **stderr** (not stdout)

- Server runs in background as subprocess

 

**Helper function:**

```python

def log_debug(message: str):

    """Logs to stderr to avoid breaking MCP JSON-RPC"""

    print(message, file=sys.stderr)

```

 

### IDE Configuration

 

**Cursor IDE:**

```json

{

  "mcpServers": {

    "a-mem": {

      "command": "python",

      "args": ["-m", "src.a_mem.main"],

      "cwd": "/absolute/path/to/a-mem-mcp-server"

    }

  }

}

```

 

**Location:**

- Windows: `%USERPROFILE%\.cursor\mcp.json`

- macOS/Linux: `~/.cursor/mcp.json`

 

### Tool Usage Patterns

 

**From AI assistant:**

```

User: "Remember this: Python uses asyncio for concurrent I/O"

 

Assistant calls:

{

  "tool": "create_atomic_note",

  "arguments": {

    "content": "Python uses asyncio for concurrent I/O",

    "source": "user_input"

  }

}

 

Response:

{

  "status": "success",

  "note_id": "732c8c3b-7c71-42a6-9534-a611b4ffe7bf",

  "message": "Note created. Evolution started in background."

}

```

 

**Retrieve:**

```

User: "What do you know about Python async?"

 

Assistant calls:

{

  "tool": "retrieve_memories",

  "arguments": {

    "query": "Python async programming",

    "max_results": 5

  }

}

 

Response:

{

  "status": "success",

  "results": [

    {

      "id": "...",

      "content": "...",

      "summary": "...",

      "type": "concept",

      "relevance_score": 0.87,

      "connected_memories": 3,

      "connected_context": "..."

    }

  ]

}

```

 

### Parallel HTTP Server

 

**When enabled** (`TCP_SERVER_ENABLED=true`):

- MCP server runs on stdio

- HTTP server runs on port 42424

- Same `MemoryController` instance shared

 

**Endpoint:**

```bash

curl http://127.0.0.1:42424/get_graph

```

 

**Use case:** External tools (visualizer, CLI) can access live graph without interfering with MCP protocol

 

---

 

## 🧪 Testing Strategy

 

### Test Structure

 

```

tests/

├── test_a_mem.py              # Core: create, retrieve, evolve

├── test_code_structure.py     # Architecture validation

├── test_new_features.py       # Type classification, priority, events

├── test_enzymes.py            # All 14+ enzymes

├── test_scheduler.py          # Automatic enzyme scheduling

├── test_mcp_integration.py    # MCP server integration

├── test_researcher*.py        # Research agent (live + mocked)

├── test_safe_graph_wrapper.py # Edge case handling

└── test_rustworkx*.py         # RustworkX backend

```

 

### Running Tests

 

**All tests:**

```bash

python tests/test_a_mem.py

python tests/test_code_structure.py

python tests/test_new_features.py

python tests/test_enzymes.py

python tests/test_scheduler.py

```

 

**Single test:**

```bash

python -m pytest tests/test_a_mem.py::test_create_note -v

```

 

**With coverage:**

```bash

pytest --cov=src/a_mem tests/

```

 

### Test Conventions

 

1. **Use temp directories** for test data

2. **Clean up** after each test (`tearDown`)

3. **Mock external services** (LLM, web requests)

4. **Test both success and error paths**

5. **Verify event logs** for critical operations

 

**Example test:**

```python

def test_create_note(self):

    note_input = NoteInput(

        content="Test note",

        source="test"

    )

    note_id = asyncio.run(self.controller.create_note(note_input))

    self.assertIsNotNone(note_id)

 

    # Verify storage

    note = asyncio.run(self.controller.get_note_data(note_id))

    self.assertEqual(note["content"], "Test note")

 

    # Verify event log

    events = self._read_events()

    self.assertTrue(any(e["event_type"] == "NOTE_CREATED" for e in events))

```

 

---

 

## 📝 Common Tasks & Patterns

 

### Task 1: Add Support for a New LLM Provider

 

1. **Update `utils/llm.py`:**

```python

def extract_metadata(self, content: str) -> dict:

    if self.provider == "ollama":

        # Existing Ollama logic

    elif self.provider == "openrouter":

        # Existing OpenRouter logic

    elif self.provider == "new_provider":

        # Your new provider logic

```

 

2. **Add config in `config.py`:**

```python

NEW_PROVIDER_API_KEY = os.getenv("NEW_PROVIDER_API_KEY", "")

NEW_PROVIDER_LLM_MODEL = os.getenv("NEW_PROVIDER_LLM_MODEL", "default-model")

```

 

3. **Update `.env.example`**

 

4. **Add tests** in `tests/test_llm.py`

 

### Task 2: Optimize Graph Performance

 

**Switch to RustworkX:**

```bash

pip install rustworkx

```

 

```bash

# .env

GRAPH_BACKEND=rustworkx

```

 

**Benchmark:**

```bash

python tools/benchmark_enzymes.py

```

 

**Expected speedup:** 3x-100x for large graphs (>1000 nodes)

 

### Task 3: Visualize Memory Graph

 

**Start visualizer:**

```bash

python tools/visualize_memory.py

```

 

**Open browser:** http://localhost:8050

 

**Features:**

- Interactive network graph (priority-based sizing, type-based coloring)

- Priority statistics by type

- Relation type distribution

- Event timeline

- Node details table

 

**Update data:**

```bash

python tools/extract_graph.py  # Requires TCP_SERVER_ENABLED=true

```

 

### Task 4: Monitor System Health

 

**Quick status:**

```bash

python tools/amem_stats.py

```

 

**Output:**

```

🧠 A-MEM Graph Status

==================================================

📝 Notes:        127

🔗 Relations:    342

📊 Notes by Type:

   🔴 rule           23

   🔵 procedure      45

   🟢 concept        59

⚙️  Last Enzyme Run: 15min ago

==================================================

```

 

**Watch mode:**

```bash

python tools/amem_stats.py --watch

```

 

**Diff mode:**

```bash

python tools/amem_stats.py --diff

# +12 notes | +28 relations | -5 zombie nodes

```

 

### Task 5: Manual Memory Maintenance

 

**Via MCP:**

```json

{

  "tool": "run_memory_enzymes",

  "arguments": {

    "prune_max_age_days": 90,

    "prune_min_weight": 0.3,

    "suggest_threshold": 0.75,

    "auto_add_suggestions": false

  }

}

```

 

**Via CLI:**

```bash

python tools/a_mem_cli.py --enzyme-run

```

 

**Schedule:**

- Automatic: Every hour (configurable in `main.py`)

- Manual: Use tool or CLI

 

---

 

## ✅ Best Practices & Conventions

 

### Code Style

 

1. **Type hints everywhere:**

```python

def my_function(param: str) -> dict:

    """Clear docstring."""

    return {}

```

 

2. **Pydantic for data validation:**

```python

class MyModel(BaseModel):

    field: str

    optional_field: Optional[int] = None

```

 

3. **Async I/O for blocking operations:**

```python

loop = asyncio.get_running_loop()

result = await loop.run_in_executor(None, blocking_func, args)

```

 

4. **Logging to stderr (not stdout):**

```python

print(message, file=sys.stderr)

```

 

### Error Handling

 

**Graceful degradation:**

```python

try:

    result = await risky_operation()

except Exception as e:

    log_debug(f"[ERROR] Operation failed: {e}")

    return {"error": str(e), "status": "partial_success"}

```

 

**Return structured errors:**

```python

return {

    "status": "error",

    "error_code": "INVALID_INPUT",

    "message": "Parameter X is required",

    "details": {...}

}

```

 

### Event Logging

 

**Always log critical operations:**

```python

from .utils.priority import log_event

 

log_event("OPERATION_NAME", {

    "key": "value",

    "timestamp": datetime.now().isoformat()

})

```

 

### Data Persistence

 

**Atomic saves:**

```python

temp_file = path.with_suffix(".tmp")

with open(temp_file, 'w') as f:

    json.dump(data, f)

temp_file.rename(path)  # Atomic on POSIX

```

 

**Backup before destructive operations:**

```python

if path.exists():

    backup = path.with_suffix(".backup")

    shutil.copy(path, backup)

```

 

### Performance Optimization

 

1. **Use RustworkX** for graphs >1000 nodes

2. **Batch operations** when possible

3. **Offload blocking I/O** to executor

4. **Cache embeddings** (already in ChromaDB)

5. **Limit graph traversal depth** (currently 1 hop)

 

---

 

## 🔍 Troubleshooting Guide

 

### Issue: MCP Server Not Starting

 

**Symptoms:**

- IDE shows "Server failed to start"

- No stderr output

 

**Solutions:**

1. Check Python path in `mcp.json`

2. Verify `cwd` is absolute path

3. Check `.env` file exists and is valid

4. Test standalone: `python -m src.a_mem.main`

5. Check logs: `tail -f data/graph_save.log`

 

### Issue: Slow Retrieval Performance

 

**Symptoms:**

- `retrieve_memories` takes >5 seconds

- Graph has >10,000 nodes

 

**Solutions:**

1. Switch to RustworkX: `GRAPH_BACKEND=rustworkx`

2. Run enzymes to prune weak edges

3. Increase `MIN_SIMILARITY_SCORE` to filter results

4. Reduce `MAX_NEIGHBORS` for less graph traversal

 

### Issue: Graph Data Lost After Restart

 

**Symptoms:**

- Notes disappear after server restart

- Graph snapshot file empty

 

**Solutions:**

1. Check file permissions on `data/graph/`

2. Verify no concurrent writes (check `graph.lock`)

3. Enable FalkorDB for persistence: `GRAPH_BACKEND=falkordb`

4. Check logs: `grep "ERROR" data/graph_save.log`

 

### Issue: Research Agent Not Working

 

**Symptoms:**

- `research_and_store` returns no notes

- Web search fails

 

**Solutions:**

1. Check environment:

   - `RESEARCHER_ENABLED=true`

   - `GOOGLE_API_KEY` set (or use DuckDuckGo fallback)

   - `JINA_READER_ENABLED=true`

2. Check server logs for `[RESEARCHER]` messages

3. Test components:

   ```bash

   python tests/test_researcher_live.py

   ```

4. Verify network access (firewalls, proxies)

 

### Issue: Memory Enzymes Causing Errors

 

**Symptoms:**

- Enzyme scheduler crashes

- Event log shows `ENZYME_ERROR`

 

**Solutions:**

1. Check enzyme parameters (age, weight, threshold)

2. Run enzymes manually with logging:

   ```bash

   python -c "from src.a_mem.utils.enzymes import *; run_memory_enzymes(...)"

   ```

3. Check for corrupted nodes:

   ```bash

   grep "CORRUPTED" data/events.jsonl

   ```

4. Reset if necessary (backup first!)

 

### Issue: Embedding Dimension Mismatch

 

**Symptoms:**

- ChromaDB error: "Embedding dimension mismatch"

- Different models produce different dimensions

 

**Solutions:**

1. Check model dimensions:

   - `nomic-embed-text`: 768

   - `text-embedding-3-small`: 1536

2. Clear ChromaDB when switching models:

   ```bash

   rm -rf data/chroma/

   ```

3. See `docs/EMBEDDING_DIMENSIONS.md`

 

---

 

## 📚 Additional Resources

 

### Documentation Files

 

- **README.md** - User-facing documentation

- **MCP_SERVER_SETUP.md** - MCP tool reference

- **docs/MEMORY_ENZYMES_DETAILED.md** - Enzyme deep dive

- **docs/RESEARCHER_AGENT_DETAILED.md** - Research agent guide

- **docs/TEST_REPORT.md** - Test results

- **docs/ARCHITECTURE_DIAGRAM.md** - Visual architecture (Mermaid)

 

### Architecture Diagrams

 

Located in `docs/*.svg`:

- `a-mem-system-architecture.svg` - Overall system

- `a-mem-storage-architecture.svg` - Storage layer

- `a-mem-memory-enzymes.svg` - Enzyme workflow

- `a-mem-mcp-tools.svg` - Tool overview

- `a-mem-type-classification.svg` - Note type system

 

### External Links

 

- [Research Paper](https://arxiv.org/html/2502.12110v11) - Original A-Mem paper

- [Original Repo](https://github.com/WujiangXu/A-mem-sys) - Authors' implementation

- [MCP Documentation](https://modelcontextprotocol.io/) - Protocol spec

 

---

 

## 🎓 Learning Path for New Contributors

 

### Week 1: Understand Architecture

1. Read this file (CLAUDE.md)

2. Read README.md

3. Explore `src/a_mem/` structure

4. Run tests: `python tests/test_a_mem.py`

5. Read `src/a_mem/models/note.py`

 

### Week 2: Core Components

1. Study `src/a_mem/core/logic.py` (MemoryController)

2. Study `src/a_mem/storage/engine.py` (Storage layer)

3. Study `src/a_mem/utils/llm.py` (LLM integration)

4. Run: `python -m src.a_mem.main` and test via Cursor

 

### Week 3: Advanced Features

1. Study `src/a_mem/utils/enzymes.py` (Memory maintenance)

2. Study `src/a_mem/utils/researcher.py` (Web research)

3. Study `src/a_mem/main.py` (MCP server)

4. Run visualizer: `python tools/visualize_memory.py`

 

### Week 4: Contribute

1. Pick an issue or feature

2. Write tests first

3. Implement feature

4. Run all tests

5. Submit PR

 

---

 

## 📞 Getting Help

 

**When asking for help, provide:**

1. Error message (full traceback)

2. Configuration (`.env` settings)

3. Steps to reproduce

4. Expected vs. actual behavior

5. Relevant logs (`data/events.jsonl`, `data/graph_save.log`)

 

**Debug checklist:**

- [ ] Check `.env` file exists and is valid

- [ ] Check Python version (3.9+)

- [ ] Check dependencies: `pip install -r requirements.txt`

- [ ] Check file permissions on `data/` directory

- [ ] Check logs for errors

- [ ] Test standalone: `python -m src.a_mem.main`

 

---

 

## 🔄 Changelog

 

**v1.0 (December 1, 2025)**

- Initial CLAUDE.md creation

- Comprehensive architecture documentation

- Development workflows and best practices

- Troubleshooting guide

- Learning path for new contributors

 

---

 

**End of CLAUDE.md**

 

> This document is maintained by the community. When making significant architectural changes, please update this file accordingly.
