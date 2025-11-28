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

### Advanced Features (New)
- ✅ **Type Classification**: Automatic classification of notes into 6 types (rule, procedure, concept, tool, reference, integration)
- ✅ **Priority Scoring**: On-the-fly priority calculation based on type, age, usage, and edge count for better search rankings
- ✅ **Event Logging**: Append-only JSONL event log for all critical operations (NOTE_CREATED, RELATION_CREATED, MEMORY_EVOLVED)
- ✅ **Memory Enzymes**: Autonomous background processes for graph maintenance
  - **Link Pruner**: Removes old/weak edges automatically
  - **Relation Suggester**: Finds new semantic connections between notes
  - **Summary Digester**: Compresses overcrowded nodes with many children
- ✅ **Automatic Scheduler**: Runs memory enzymes every hour in the background
- ✅ **Metadata Field**: Experimental fields support without schema changes

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
  - NetworkX DiGraph for explicit typed relationships (with `relation_type`, `reasoning`, `weight`)
  - Graph traversal for finding directly connected memories
  - Enables complex queries like "find all memories related to X through type Y"

- **Original implementation**: Single-storage architecture
  - ChromaDB as primary storage
  - Implicit linking through embedding similarity
  - Simpler architecture, less overhead

Both implementations are valid approaches to the same research paper, serving different use cases and integration scenarios.

## 🏗️ Framework

The framework of our Agentic Memory system showing the dynamic interaction between LLM agents and memory components, extended with our MCP Server integration and dual-storage architecture:

![Framework](framework_extended.jpg)

*The framework diagram illustrates the core memory system workflow: Note Construction (left), Memory Processing with MCP Server Integration (center), and Memory Retrieval (right). Our implementation extends the original framework with direct IDE integration via MCP protocol, explicit graph-based memory linking using NetworkX DiGraph with typed edges (relation_type, reasoning, weight), file import with automatic chunking, and a dual-storage architecture using ChromaDB for vector similarity search and NetworkX for explicit typed relationships.*

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
- `RELATIONS_SUGGESTED`: When new connections are found
- `ENZYME_SCHEDULER_RUN`: When automatic maintenance runs

### Memory Enzymes
Autonomous background processes that maintain graph health:
- **Link Pruner**: Removes edges older than 90 days or with weight < 0.3, and orphaned edges to missing/zombie nodes
- **Zombie Node Remover**: Automatically removes nodes without content (empty nodes)
- **Relation Suggester**: Finds semantically similar notes (cosine similarity ≥ 0.75)
- **Summary Digester**: Compresses nodes with >8 children into compact summaries

### Automatic Scheduler
The system automatically runs memory enzymes every hour:
- Runs in background without blocking MCP operations
- Logs all maintenance activities
- Gracefully handles errors and continues running

## 📋 Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and adjust the values:

```bash
cp .env.example .env
```

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

### Available Tools (14 Total)

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
14. **`run_memory_enzymes`** - Runs memory maintenance: prunes old/weak links and zombie nodes, suggests new relations, digests overcrowded nodes. Automatically optimizes graph structure

### IDE Integration

#### Cursor IDE

1. Open the MCP configuration file:
   - Windows: `%USERPROFILE%\.cursor\mcp.json` (or `C:\Users\<username>\.cursor\mcp.json`)
   - macOS: `~/.cursor/mcp.json`
   - Linux: `~/.cursor/mcp.json`

2. Add the following configuration:

```json
{
  "mcpServers": {
    "a-mem": {
      "command": "python",
      "args": [
        "-m",
        "src.a_mem.main"
      ],
      "cwd": "/path/to/a-mem-mcp-server"
    }
  }
}
```

**Important:** Adjust `cwd` to the absolute path to your project directory!

3. Restart Cursor to load the configuration.

#### Visual Studio Code (with MCP Extension)

1. Install the MCP Extension for VSCode (if available)

2. Open VSCode Settings (JSON):
   - `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS)
   - Type "Preferences: Open User Settings (JSON)"

3. Add the MCP Server configuration:

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

**Alternative:** Use the `mcp.json` file in the project root:

```json
{
  "mcpServers": {
    "a-mem": {
      "command": "python",
      "args": ["-m", "src.a_mem.main"],
      "cwd": "${workspaceFolder}"
    }
  }
}
```

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
✅ **MCP Server Integration** (14 Tools)  
✅ **Memory Reset & Management Tools**  
✅ **Type Classification & Priority Scoring**  
✅ **Event Logging & Audit Trail**  
✅ **Memory Enzymes (Autonomous Graph Maintenance)**  
✅ **Automatic Scheduler (Hourly Maintenance)**  
✅ **HTTP Server** (optional, for external tools like `extract_graph.py`)

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
