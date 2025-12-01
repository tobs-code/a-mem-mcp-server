# A-MEM: Agentic Memory System

An agentic memory system for LLM agents based on the Zettelkasten principle.

> **Based on:** ["A-Mem: Agentic Memory for LLM Agents"](https://arxiv.org/html/2502.12110v11)  
> by Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, Yongfeng Zhang  
> Rutgers University, Independent Researcher, AIOS Foundation

## 🚀 Features

### Core Features
- ✅ **Note Construction**: Automatic extraction of keywords, tags, and contextual summary
- ✅ **Link Generation**: Automatic linking of similar memories
- ✅ **Memory Evolution**: Dynamic updating of existing memories
- ✅ **Semantic Retrieval**: Intelligent search with graph traversal
- ✅ **Multi-Provider Support**: Ollama (local) or OpenRouter (cloud)
- ✅ **Environment Variables**: Configuration via `.env` file
- ✅ **Graph Backend Selection**: Choose between NetworkX (default), RustworkX (3x-100x faster), or FalkorDB (experimental, not fully tested)
- ✅ **Parameter Validation**: Automatic validation of all MCP tool parameters
- ✅ **Safe Graph Wrapper**: Edge case handling and data sanitization for robust operations

### Advanced Features (New)
- ✅ **Type Classification**: Automatic classification of notes into 6 types (rule, procedure, concept, tool, reference, integration)
- ✅ **Priority Scoring**: On-the-fly priority calculation based on type, age, usage, and edge count for better search rankings
- ✅ **Event Logging**: Append-only JSONL event log for all critical operations (NOTE_CREATED, RELATION_CREATED, MEMORY_EVOLVED)
- ✅ **Memory Enzymes**: Autonomous background processes for graph maintenance
  - **Link Pruner**: Removes old/weak edges automatically
  - **Relation Suggester**: Finds new semantic connections between notes
  - **Dead-End Node Auto-Linker**: Automatically creates out-connections for dead-end nodes with intelligent filtering
  - **Summary Digester**: Compresses overcrowded nodes with many children
- ✅ **Automatic Scheduler**: Runs memory enzymes every hour in the background
- ✅ **Metadata Field**: Experimental fields support without schema changes
- ✅ **Researcher Agent**: Deep web research for low-confidence queries (JIT context optimization)
  - Automatic triggering when retrieval confidence < threshold
  - Manual research via `research_and_store` MCP tool
  - Hybrid approach: MCP tools (if available) or HTTP-based fallbacks (Google Search API, DuckDuckGo, Jina Reader)
- ✅ **Local Jina Reader**: Support for local Docker-based Jina Reader instance (fallback to cloud API)
- ✅ **Unstructured PDF Extraction**: Automatic PDF extraction using Unstructured (library or API)

## 🔄 Relationship to Original Implementation

This implementation was developed independently based on the research paper ["A-Mem: Agentic Memory for LLM Agents"](https://arxiv.org/html/2502.12110v11). The original authors' production-ready system ([A-mem-sys](https://github.com/WujiangXu/A-mem-sys)) was discovered after this implementation was completed.

**Key Differences:**

This implementation focuses on **MCP Server integration** for IDE environments (Cursor, VSCode), providing:
- Direct IDE integration via MCP protocol
- **Explicit graph-based memory linking** using NetworkX (DiGraph) with typed edges, reasoning, and weights
- File import with automatic chunking
- Memory reset and management tools
- Modern TUI benchmarking tool for Ollama model speed testing

The original [A-mem-sys](https://github.com/WujiangXu/A-mem-sys) repository provides a **pip-installable Python library** with:
- Multiple LLM backend support (OpenAI, Ollama, OpenRouter, SGLang)
- Library-based integration for Python applications
- Comprehensive API for programmatic usage
- **Implicit linking** via ChromaDB embeddings (no explicit graph structure)

**Technical Architecture Difference:**

- **This implementation**: Dual-storage architecture
  - ChromaDB for vector similarity search
  - **Graph Backend Selection**: NetworkX (default), RustworkX (3x-100x faster), or FalkorDB (experimental, not fully tested)
  - Explicit typed relationships (with `relation_type`, `reasoning`, `weight`)
  - Graph traversal for finding directly connected memories
  - Enables complex queries like "find all memories related to X through type Y"
  - **Safe Graph Wrapper**: Automatic edge case handling and data sanitization

- **Original implementation**: Single-storage architecture
  - ChromaDB as primary storage
  - Implicit linking through embedding similarity
  - Simpler architecture, less overhead

Both implementations are valid approaches to the same research paper, serving different use cases and integration scenarios.

## 🆕 New Features Overview

### Type Classification
Every note is automatically classified into one of 6 types:
- **rule**: Imperative instructions ("Never X", "Always Y")
- **procedure**: Numbered steps or sequential instructions
- **concept**: Explanations of concepts, no commands
- **tool**: Describes functions, APIs, or utilities
- **reference**: Tables, comparison lists, cheatsheets
- **integration**: Describes connections between systems

### Priority Scoring
Search results are ranked using on-the-fly priority calculation:
- **Type Weight**: Rules and procedures have higher priority
- **Age Factor**: Newer notes have higher priority
- **Usage Count**: Frequently accessed notes get boosted
- **Edge Count**: Well-connected notes are prioritized

### Event Logging
All critical operations are logged to `data/events.jsonl`:
- `NOTE_CREATED`: When a new note is created
- `RELATION_CREATED`: When two notes are linked
- `MEMORY_EVOLVED`: When an existing note is updated
- `LINKS_PRUNED`: When old/weak links are removed
- `RELATION_PRUNED`: When a specific relation is pruned (with reason)
- `NODE_PRUNED`: When a zombie node is removed
- `DUPLICATES_MERGED`: When duplicate notes are merged
- `SELF_LOOPS_REMOVED`: When self-loops are removed
- `ISOLATED_NODES_FOUND`: When isolated nodes are detected
- `ISOLATED_NODES_LINKED`: When isolated nodes are automatically linked
- `KEYWORDS_NORMALIZED`: When keywords are normalized
- `QUALITY_SCORES_CALCULATED`: When quality scores are calculated
- `NOTES_VALIDATED`: When notes are validated
- `TYPES_VALIDATED`: When note types are validated and corrected
- `NOTES_ARCHIVED`: When old notes are archived (temporal cleanup)
- `NOTES_DELETED`: When old notes are deleted (temporal cleanup)
- `GRAPH_HEALTH_CALCULATED`: When graph health score is calculated
- `DEAD_END_NODES_FOUND`: When dead-end nodes are detected
- `DEAD_END_NODES_AUTO_LINKED`: When dead-end nodes are automatically linked with out-connections
- `LOW_QUALITY_NOTES_REMOVED`: When low-quality notes are removed
- `CORRUPTED_NODES_REPAIRED`: When corrupted nodes are repaired
- `RELATIONS_SUGGESTED`: When new connections are found
- `ENZYME_SCHEDULER_RUN`: When automatic maintenance runs

### Memory Enzymes
Autonomous background processes that maintain graph health:
- **Link Pruner**: Removes edges older than 90 days or with weight < 0.3, and orphaned edges to missing/zombie nodes
- **Zombie Node Remover**: Automatically removes nodes without content (empty nodes)
- **Duplicate Merger**: Finds and merges duplicate notes (exact matches + semantic duplicates via embeddings)
- **Edge Validator**: Validates and fixes edges (adds missing reasoning, standardizes types, removes weak edges)
- **Self-Loop Remover**: Removes self-referential edges (nodes linking to themselves)
- **Isolated Node Finder**: Identifies nodes without any connections
- **Isolated Node Linker**: Automatically links isolated nodes to similar notes (similarity threshold: 0.70)
- **Keyword Normalizer**: Normalizes and cleans keywords (removes duplicates, corrects typos, limits to max 7 keywords)
- **Quality Score Calculator**: Calculates quality scores for notes (based on content, metadata, connections)
- **Note Validator**: Validates notes and corrects missing/invalid fields (summary, keywords, tags)
- **Type Validator**: Validates and corrects invalid note types using LLM-based classification
- **Temporal Note Cleanup**: Archives or deletes notes older than specified age (default: 365 days)
- **Graph Health Score Calculator**: Calculates overall graph health score (0.0-1.0) based on average note quality, connectivity, edge quality, and completeness
- **Dead-End Node Detector**: Identifies nodes with incoming but no outgoing edges (dead ends in knowledge flow)
- **Dead-End Node Auto-Linker**: Automatically creates out-connections for dead-end nodes using intelligent filtering:
  - **Combined Strategy**: Ignores generic tags (reference, documentation, tool, etc.), prioritizes keywords, requires similarity check
  - **Smart Thresholds**: 0.46 for strong signals (keywords or meaningful tags), 0.60 for weak signals (generic tags only), 0.58 for no signals
  - **Quality Control**: Prevents low-quality connections by requiring meaningful semantic signals
- **Low Quality Note Remover**: Removes irrelevant notes (CAPTCHA pages, error pages, spam)
- **Summary Refiner**: Refines similar summaries to make them more specific and distinct
- **Corrupted Node Repairer**: Repairs corrupted nodes (missing fields, invalid data)
- **Relation Suggester**: Finds semantically similar notes (cosine similarity ≥ 0.75) with dead-end node prioritization
- **Summary Digester**: Compresses nodes with >8 children into compact summaries

### Automatic Scheduler
The system automatically runs memory enzymes every hour:
- Runs in background without blocking MCP operations
- Executes 19+ maintenance operations in optimized sequence
- Logs all maintenance activities with detailed metrics
- Gracefully handles errors and continues running
- **Comprehensive Results**: Returns detailed statistics for all operations (pruned links, merged duplicates, validated notes, quality scores, graph health score, dead-end nodes, etc.)

### Researcher Agent
Deep web research for low-confidence queries with JIT context optimization:
- **Automatic Triggering**: Activates when retrieval confidence < threshold (default: 0.5)
- **Manual Research**: Available via `research_and_store` MCP tool
- **Hybrid Approach**: Uses MCP tools (if available) or HTTP-based fallbacks
- **Web Search**: Google Search API (primary) or DuckDuckGo HTTP (fallback)
- **Content Extraction**: Jina Reader (local Docker or cloud API) or Readability fallback
- **PDF Support**: Automatic PDF extraction using Unstructured (library or API)
- **Automatic Note Creation**: Research findings are automatically stored as atomic notes with metadata, keywords, and tags

## 📋 Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### Graph Backend Selection

A-MEM supports three graph backends, selectable via `GRAPH_BACKEND` environment variable:

**NetworkX (Default):**
- ✅ Included by default (no extra installation)
- ✅ Cross-platform (Windows, Linux, macOS)
- ✅ Good for small to medium graphs (<10k nodes)

**RustworkX (Recommended for Performance):**
- ⚡ **3x-100x faster** than NetworkX
- ✅ Windows-compatible
- ✅ Install with: `pip install rustworkx`
- ✅ Set `GRAPH_BACKEND=rustworkx` in `.env`

**FalkorDB (Experimental - Not Fully Integrated/Tested):**
- ⚠️ **Proof-of-Concept Status** - Funktional, aber noch nicht vollständig integriert und getestet
- 💾 **Persistent storage** (survives restarts)
- ⚠️ **Nicht empfohlen für Production** - Siehe `docs/FALKORDB_POC_README.md` für Details
- **Linux/macOS**: Install with `pip install falkordblite`, set `GRAPH_BACKEND=falkordb`
- **Windows**: Install with `pip install falkordb redis`, requires Redis with FalkorDB module (see `docs/WINDOWS_FALKORDB_SETUP.md`)
- ⚠️ **Bekannte Einschränkungen**: Memory Enzymes nutzen noch direkte `graph.graph` Zugriffe, Migration-Tool fehlt, Performance noch nicht getestet

**Safe Graph Wrapper:**
- 🛡️ Automatic edge case handling and data sanitization
- ✅ Enabled automatically when using RustworkX
- ✅ Validates and sanitizes all graph operations

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and adjust the values:

```bash
# Windows (PowerShell)
Copy-Item .env.example .env

# Linux/macOS
cp .env.example .env
```

**Important:** The `.env.example` file contains all available configuration options including the `GRAPH_BACKEND` setting for selecting the graph database backend.

**Configuration Strategy (Hybrid Approach):**

A-MEM uses a **hybrid configuration system** for maximum flexibility:

- **`.env` file**: Base configuration (default settings for all instances)
- **MCP `env` block**: Per-instance overrides (allows customization without modifying `.env`)
- **Priority**: MCP `env` block values override `.env` file values

This allows you to:
- Keep default settings in `.env` for all instances
- Override specific settings per MCP instance (e.g., different LLM models for different projects)
- Share the same `.env` across multiple MCP configurations

**Configuration:**

- **LLM_PROVIDER**: `"ollama"` (local) or `"openrouter"` (cloud)
- **Ollama**: Local models (default)
- **OpenRouter**: Cloud-based LLMs (requires API key)

**Example `.env` for Ollama (default):**
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=qwen3:4b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text:latest
```

**Example `.env` for OpenRouter:**
```env
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_LLM_MODEL=openai/gpt-4o-mini
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
```

**Graph Backend Selection (Optional):**
```env
# Default: networkx (no installation needed)
GRAPH_BACKEND=networkx

# For better performance (3x-100x faster):
# pip install rustworkx
GRAPH_BACKEND=rustworkx

# For persistent storage (production):
# Experimental - Not fully tested (see docs/FALKORDB_POC_README.md)
# Linux/macOS: pip install falkordblite
# Windows: pip install falkordb redis (see docs/WINDOWS_FALKORDB_SETUP.md)
# GRAPH_BACKEND=falkordb  # ⚠️ Experimental - Use at your own risk
```

**Researcher Agent & Content Extraction (Optional):**

```env
# Researcher Agent (for low-confidence retrieval)
RESEARCHER_ENABLED=true
RESEARCHER_CONFIDENCE_THRESHOLD=0.5
RESEARCHER_MAX_SOURCES=5
RESEARCHER_MAX_CONTENT_LENGTH=10000

# Google Search API (for web search) - uses existing GetWeb config
GOOGLE_SEARCH_ENABLED=true
GOOGLE_API_KEY=your_google_api_key  # Optional, uses default if not set
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id  # Optional, uses default if not set

# Local Jina Reader (Docker) - for web content extraction
# If you have a local Jina Reader Docker instance running
JINA_READER_ENABLED=true
JINA_READER_HOST=localhost
JINA_READER_PORT=2222

# Unstructured (for PDF extraction)
# Option 1: Use library directly (requires: pip install unstructured[pdf])
UNSTRUCTURED_ENABLED=true
UNSTRUCTURED_USE_LIBRARY=true

# Option 2: Use API (if Unstructured API is running)
# UNSTRUCTURED_ENABLED=true
# UNSTRUCTURED_API_URL=http://localhost:8000
# UNSTRUCTURED_API_KEY=your_api_key_here  # Optional
```

**Note:** 
- **Web Search**: The Researcher Agent uses **Google Search API** (if configured) for high-quality search results, falling back to DuckDuckGo HTTP search if not available.
- **Content Extraction**: Uses **local Jina Reader** (if enabled) for web content extraction, falling back to cloud API if local instance is unavailable.
- **PDF Extraction**: For **PDF URLs**, the Researcher Agent uses **Unstructured** for extraction (library or API).
- **Extraction Strategy**: 
  - Search: Google Search API → DuckDuckGo HTTP fallback
  - Web Content: Jina Reader (local/cloud) → Readability fallback
  - PDFs: Unstructured (library/API)

**Installing Unstructured for PDF extraction:**
```bash
# Full installation with PDF support
pip install "unstructured[pdf]"

# Or minimal installation (may require additional dependencies)
pip install unstructured
pip install pdfminer.six  # Required for PDF extraction
```
**Optional HTTP Server (for external tools):**
```env
TCP_SERVER_ENABLED=true
TCP_SERVER_HOST=127.0.0.1
TCP_SERVER_PORT=42424
```

When enabled, the MCP server runs an additional HTTP server in parallel, exposing the graph data at `http://127.0.0.1:42424/get_graph`. This allows external tools (like `extract_graph.py`) to access the current graph state without interfering with the stdio MCP protocol.

### 3. Install Ollama Models (only when LLM_PROVIDER=ollama)

```bash
ollama pull qwen3:4b
ollama pull nomic-embed-text:latest
```

### 4. Start Ollama (only when LLM_PROVIDER=ollama)

Make sure Ollama is running on `http://localhost:11434`.

## 🛠️ MCP Server

### Start

```bash
python mcp_server.py
```

### Available Tools (15 Total)

#### Core Memory Operations
1. **`create_atomic_note`** - Stores a new piece of information. Automatically classifies note type, extracts metadata, and starts linking/evolution in background
2. **`retrieve_memories`** - Searches for relevant memories with priority scoring. Returns best matches ranked by combined similarity and priority
3. **`get_memory_stats`** - Returns statistics about the memory system (nodes, edges, etc.)
4. **`add_file`** - Stores file content as notes with automatic chunking for large files (>16KB)
5. **`reset_memory`** - Resets the complete memory system (⚠️ irreversible)

#### Note Management
6. **`list_notes`** - Lists all stored notes from the memory graph
7. **`get_note`** - Returns a single note (metadata + content) by ID
8. **`update_note`** - Updates contextual summary, tags, or keywords for an existing note
9. **`delete_atomic_note`** - Deletes a note and all associated connections

#### Relation Management
10. **`list_relations`** - Lists relations in the graph, optionally filtered by note ID
11. **`add_relation`** - Adds a manual relation between two notes
12. **`remove_relation`** - Removes a relation between two notes

#### Graph Operations
13. **`get_graph`** - Returns the full graph snapshot (nodes + edges) for visualization

#### Memory Maintenance
14. **`run_memory_enzymes`** - Runs comprehensive memory maintenance: prunes old/weak links and zombie nodes, merges duplicates, validates and fixes edges, removes self-loops, links isolated nodes, normalizes keywords, calculates quality scores, validates notes and types, performs temporal cleanup, calculates graph health score, detects dead-end nodes, automatically creates out-connections for dead-end nodes (with intelligent filtering), removes low-quality content, refines summaries, repairs corrupted nodes, suggests new relations, and digests overcrowded nodes. Automatically optimizes graph structure with 19+ maintenance operations

#### Research & Web Integration
15. **`research_and_store`** - Performs deep web research on a query and stores findings as atomic notes. Uses Google Search API (if configured) or DuckDuckGo HTTP search, extracts content with Jina Reader (local Docker or cloud), and processes PDFs with Unstructured. Automatically creates notes with metadata, keywords, and tags.

### IDE Integration

#### Cursor IDE

1. Open the MCP configuration file:
   - Windows: `%USERPROFILE%\.cursor\mcp.json` (or `C:\Users\<username>\.cursor\mcp.json`)
   - macOS: `~/.cursor/mcp.json`
   - Linux: `~/.cursor/mcp.json`

2. Add the following configuration:

**Basic Configuration (uses `.env` file):**

```json
{
  "mcpServers": {
    "a-mem": {
      "command": "python",
      "args": [
        "-m",
        "src.a_mem.main"
      ],
      "cwd": "C:\\Users\\tobs\\Downloads\\a-mem_-agentic-memory-system\\a-mem_-agentic-memory-system"
    }
  }
}
```

**Advanced Configuration (with environment overrides):**

You can override `.env` settings directly in the MCP config using the `env` block. This allows per-instance customization without modifying the `.env` file:

```json
{
  "mcpServers": {
    "a-mem": {
      "command": "python",
      "args": [
        "-m",
        "src.a_mem.main"
      ],
      "cwd": "C:\\Users\\tobs\\Downloads\\a-mem_-agentic-memory-system\\a-mem_-agentic-memory-system",
      "env": {
        "LLM_PROVIDER": "ollama",
        "OLLAMA_LLM_MODEL": "qwen3:4b",
        "RESEARCHER_ENABLED": "true",
        "RESEARCHER_CONFIDENCE_THRESHOLD": "0.5"
      }
    }
  }
}
```

**Configuration Strategy:**
- **`.env` file**: Default configuration (loaded automatically)
- **`env` block**: Overrides specific settings per MCP instance
- **Priority**: `env` block values override `.env` file values

**Important:** Adjust `cwd` to the absolute path to your project directory!

3. Restart Cursor to load the configuration.

#### Visual Studio Code (with MCP Extension)

1. Install the MCP Extension for VSCode (if available)

2. Open VSCode Settings (JSON):
   - `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS)
   - Type "Preferences: Open User Settings (JSON)"

3. Add the MCP Server configuration:

**Basic Configuration (uses `.env` file):**

```json
{
  "mcp.servers": {
    "a-mem": {
      "command": "python",
      "args": ["-m", "src.a_mem.main"],
      "cwd": "/path/to/a-mem-mcp-server"
    }
  }
}
```

**Advanced Configuration (with environment overrides):**

```json
{
  "mcp.servers": {
    "a-mem": {
      "command": "python",
      "args": ["-m", "src.a_mem.main"],
      "cwd": "/path/to/a-mem-mcp-server",
      "env": {
        "LLM_PROVIDER": "ollama",
        "OLLAMA_LLM_MODEL": "qwen3:4b",
        "RESEARCHER_ENABLED": "true"
      }
    }
  }
}
```

**Alternative:** Use the `mcp.json` file in the project root:

```json
{
  "mcpServers": {
    "a-mem": {
      "command": "python",
      "args": ["-m", "src.a_mem.main"],
      "cwd": "${workspaceFolder}",
      "env": {
        "LLM_PROVIDER": "ollama"
      }
    }
  }
}
```

**Note:** The `env` block allows you to override `.env` settings per-instance without modifying the base configuration file.

#### Usage in IDE

After configuration, the MCP tools are directly available in your IDE:

- **Chat/Composer**: Use the tools via natural language
  - "Store this information: ..."
  - "Search for memories about: ..."
  - "Show me the memory statistics"

- **Code**: The tools are automatically available as functions

See `MCP_SERVER_SETUP.md` for detailed information about all available tools.

### Event Log

All system events are automatically logged to `data/events.jsonl` in JSONL format (one JSON object per line). This provides a complete audit trail of:
- Note creation and updates
- Relation creation and removal
- Memory evolution events
- Enzyme maintenance runs
- Scheduler activities

You can view the event log with:
```bash
# View last 10 events
tail -n 10 data/events.jsonl

# View all events
cat data/events.jsonl | jq .
```

## 📚 Documentation

- `MCP_SERVER_SETUP.md` - MCP Server Setup and Configuration
- `docs/TEST_REPORT.md` - Test Results
- `docs/MCP_SERVER_TEST_REPORT.md` - MCP Server Integration Tests
- `docs/EMBEDDING_DIMENSIONS.md` - Embedding Dimension Handling Guide
- `docs/RESEARCHER_AGENT_DETAILED.md` - **Ausführliche Researcher Agent Dokumentation** (Code-Einbindung, Funktionen, Workflow)
- `docs/MEMORY_ENZYMES_DETAILED.md` - **Ausführliche Memory Enzymes Dokumentation** (Code-Einbindung, Funktionen, Workflow)
- `docs/ARCHITECTURE_DIAGRAM.md` - **Vollständige Architektur-Darstellung** (Mermaid Diagramme)
- `tools/RESEARCHER_README.md` - Researcher Agent Quick Reference

### 📊 Visual Diagrams

Aktualisierte Mermaid-basierte SVG-Diagramme mit weißem Hintergrund und schwarzer Schrift:

#### System-Architektur Übersicht
<img src="docs/a-mem-system-architecture.svg" alt="System Architecture" width="3200" style="max-width: 100%; height: auto;">

#### Storage-Architektur Detail
<img src="docs/a-mem-storage-architecture.svg" alt="Storage Architecture" width="3200" style="max-width: 100%; height: auto;">

#### Memory Enzymes Workflow
<img src="docs/a-mem-memory-enzymes.svg" alt="Memory Enzymes" width="3200" style="max-width: 100%; height: auto;">

#### MCP Tools Übersicht
<img src="docs/a-mem-mcp-tools.svg" alt="MCP Tools" width="3200" style="max-width: 100%; height: auto;">

#### Type Classification System
<img src="docs/a-mem-type-classification.svg" alt="Type Classification" width="3200" style="max-width: 100%; height: auto;">

**Vollständige Mermaid-Diagramme** (inkl. Sequence Diagrams, Workflows) finden Sie in:
- `docs/ARCHITECTURE_DIAGRAM.md` - Vollständige Architektur-Darstellung mit allen Diagrammen
## 🧪 Tests

```bash
# Core functionality tests
python tests/test_a_mem.py

# Code structure tests
python tests/test_code_structure.py

# New features tests (Type Classification, Priority Scoring, Event Logging)
python tests/test_new_features.py

# Memory enzymes tests (Link Pruner, Relation Suggester, Digest Node)
python tests/test_enzymes.py

# Scheduler tests
python tests/test_scheduler.py
```

**Test Results:** 24/24 tests passed ✅

## 🧪 Benchmarking

The project includes a modern TUI benchmark tool for testing Ollama model speed and performance:

```bash
python ollama_benchmark.py
```

This tool measures model speed metrics (tokens/sec, latency, first token time) to help you choose the best Ollama model for your use case.

See `BENCHMARK_README.md` for details.

## 📊 Graph Visualization

The project includes a web-based dashboard for visualizing the memory graph, analyzing priorities, and exploring patterns:

```bash
python tools/visualize_memory.py
```

Then open your browser to `http://localhost:8050` to view the interactive dashboard.

**Features:**
- **Graph Visualization**: Interactive network graph with node sizes based on priority and colors based on type
- **Priority Statistics**: Box plots showing priority distribution by note type
- **Relations Analysis**: Bar chart of relation types distribution
- **Event Timeline**: Timeline visualization of all system events
- **Node Details**: Detailed table with priority, edge count, summaries, and tags

The dashboard automatically refreshes when you click the refresh button, allowing you to explore patterns and insights in your memory system.

**Data Sync:** The visualizer loads graph data from `data/graph/knowledge_graph.json`. To update the data, run:
```bash
python tools/extract_graph.py
```

This script connects to the running MCP server via HTTP (if `TCP_SERVER_ENABLED=true` in `.env`) and saves the current graph state to disk.

## 📈 Live Graph Status

Quick status check for your memory system (similar to `git status`):

```bash
# Full status (default)
python tools/amem_stats.py

# Compact one-line output
python tools/amem_stats.py --compact

# JSON output (for scripting)
python tools/amem_stats.py --json

# Show changes since last run
python tools/amem_stats.py --diff

# Watch mode (auto-refresh every 5 seconds)
python tools/amem_stats.py --watch

# Watch mode with custom interval (10 seconds)
python tools/amem_stats.py --watch 10
```

**Output Examples:**

**Full Status:**
```
🧠 A-MEM Graph Status
==================================================
📝 Notes:        47
🔗 Relations:    89
📊 Notes by Type:
   🔴 rule           12
   🔵 procedure      15
   🟢 concept        20
⚙️  Last Enzyme Run: 23min ago
📡 Data Source: http
==================================================
```

**Compact Mode:**
```
47 notes | 89 relations | Last: 23min ago
```

**Diff Mode:**
```
+5 notes | +12 relations | -3 zombie nodes
```

**Watch Mode:**
Continuously refreshes the display every N seconds. Perfect for monitoring while coding! Press `Ctrl+C` to stop.

The tool automatically tries to fetch live data from the running MCP server (if HTTP server is enabled), otherwise reads from disk. Perfect for monitoring your memory system while coding!

**Shell Alias (Optional):**
Add to your `~/.bashrc`, `~/.zshrc`, or PowerShell profile:
```bash
alias amem="python ~/path/to/a-mem-mcp-server/tools/amem_stats.py"
# Then simply: amem
```

## 📊 Status

✅ **100% Paper-Compliance**  
✅ **All Tests Passed** (24/24 tests)  
✅ **Modular Structure**  
✅ **Multi-Provider Support** (Ollama + OpenRouter)  
✅ **MCP Server Integration** (15 Tools)  
✅ **Memory Reset & Management Tools**  
✅ **Type Classification & Priority Scoring**  
✅ **Event Logging & Audit Trail**  
✅ **Memory Enzymes (Autonomous Graph Maintenance)**  
✅ **Automatic Scheduler (Hourly Maintenance)**  
✅ **HTTP Server** (optional, for external tools like `extract_graph.py`)
✅ **Graph Backend Selection** (NetworkX, RustworkX, FalkorDB - experimental)
✅ **Parameter Validation** (automatic validation of all MCP tool parameters)
✅ **Safe Graph Wrapper** (edge case handling and data sanitization)
⚠️ **Windows-FalkorDB Support** (experimental, dedicated adapter for Windows users - not fully tested)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This implementation is based on the research paper ["A-Mem: Agentic Memory for LLM Agents"](https://arxiv.org/html/2502.12110v11).

## 🙏 Acknowledgments

- Original paper authors: Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, Yongfeng Zhang
- Original repositories:
  - [AgenticMemory](https://github.com/WujiangXu/AgenticMemory) - Benchmark Evaluation
  - [A-mem-sys](https://github.com/WujiangXu/A-mem-sys) - Production-ready System

---

**Created by tobi and the CURSOR IDE with the new Composer 1 model for the community ❤️**
