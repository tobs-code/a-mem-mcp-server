"""
MCP Server for A-MEM: Agentic Memory System

Complete MCP Server with tools for Memory Management.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from .core.logic import MemoryController
from .models.note import NoteInput
from .utils.enzymes import run_memory_enzymes
from .config import settings

# Helper function to print to stderr (MCP uses stdout for JSON-RPC)
def log_debug(message: str):
    """Logs debug messages to stderr to avoid breaking MCP JSON-RPC on stdout."""
    print(message, file=sys.stderr)

# Server initialisieren
server = Server("a-mem")
controller = MemoryController()

@server.list_tools()
async def list_tools() -> list[Tool]:
    """Lists all available tools."""
    return [
        Tool(
            name="create_atomic_note",
            description="Stores a new piece of information in the memory system. Automatically classifies the note type (rule, procedure, concept, tool, reference, integration), extracts metadata, and starts the linking and evolution process in the background.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The text of the note/memory to be stored."
                    },
                    "source": {
                        "type": "string",
                        "description": "Source of the information (e.g., 'user_input', 'file', 'api').",
                        "default": "user_input"
                    }
                },
                "required": ["content"]
            }
        ),
        Tool(
            name="retrieve_memories",
            description="Searches for relevant memories based on semantic similarity with priority scoring. Returns the best matches with linked contexts, ranked by combined similarity and priority score.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for the memory search."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5).",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_memory_stats",
            description="Returns statistics about the memory system (number of nodes, edges, etc.).",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="add_file",
            description="Stores the content of a file (e.g., .md) as a note in the memory system. Supports automatic chunking for large files (>16KB). Note: Requires an absolute path or the file must be in the server directory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to be stored. Must be an absolute path or the file must be located in the server directory."
                    },
                    "file_content": {
                        "type": "string",
                        "description": "Alternatively: Direct file content as string (when file_path is not provided)."
                    },
                    "chunk_size": {
                        "type": "integer",
                        "description": "Maximum size per chunk in bytes (default: 15000, to stay under 16KB limit).",
                        "default": 15000,
                        "minimum": 1000,
                        "maximum": 16384
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="reset_memory",
            description="Resets the complete memory system (Graph + Vector Store). Deletes all notes, edges, and embeddings. WARNING: This action cannot be undone!",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="list_notes",
            description="Lists all stored notes from the memory graph.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_note",
            description="Returns a single note (metadata + content) by id.",
            inputSchema={
                "type": "object",
                "properties": {
                    "note_id": {
                        "type": "string",
                        "description": "The UUID of the note."
                    }
                },
                "required": ["note_id"]
            }
        ),
        Tool(
            name="update_note",
            description="Updates contextual summary, tags or keywords for an existing note.",
            inputSchema={
                "type": "object",
                "properties": {
                    "note_id": {"type": "string"},
                    "data": {
                        "type": "object",
                        "description": "Fields to update (contextual_summary, tags, keywords)."
                    }
                },
                "required": ["note_id", "data"]
            }
        ),
        Tool(
            name="delete_atomic_note",
            description="Deletes a note from the memory system. Removes the note from Graph and Vector Store as well as all associated connections.",
            inputSchema={
                "type": "object",
                "properties": {
                    "note_id": {
                        "type": "string",
                        "description": "The UUID of the note to be deleted."
                    }
                },
                "required": ["note_id"]
            }
        ),
        Tool(
            name="list_relations",
            description="Lists relations in the memory graph, optionally filtered by note id.",
            inputSchema={
                "type": "object",
                "properties": {
                    "note_id": {
                        "type": "string",
                        "description": "Optional note id filter"
                    }
                }
            }
        ),
        Tool(
            name="add_relation",
            description="Adds a manual relation between two notes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "target_id": {"type": "string"},
                    "relation_type": {"type": "string"},
                    "reasoning": {"type": "string"},
                    "weight": {"type": "number"}
                },
                "required": ["source_id", "target_id"]
            }
        ),
        Tool(
            name="remove_relation",
            description="Removes a relation between two notes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "target_id": {"type": "string"}
                },
                "required": ["source_id", "target_id"]
            }
        ),
        Tool(
            name="get_graph",
            description="Returns the full graph snapshot (nodes + edges). Optionally saves the graph to disk for visualization tools.",
            inputSchema={
                "type": "object",
                "properties": {
                    "save": {
                        "type": "boolean",
                        "description": "If true, saves the graph to disk before returning (useful for visualization tools).",
                        "default": False
                    }
                }
            }
        ),
        Tool(
            name="run_memory_enzymes",
            description="Runs comprehensive memory maintenance enzymes: repairs corrupted nodes, prunes old/weak links and zombie nodes, validates and corrects note types/keywords/tags, refines duplicate summaries, suggests new relations, links isolated nodes, digests overcrowded nodes, performs temporal cleanup (archives old notes), calculates graph health score, and detects dead-end nodes. Automatically optimizes the entire graph structure.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prune_max_age_days": {
                        "type": "integer",
                        "description": "Maximum age in days for edges to be pruned (default: 90).",
                        "default": 90
                    },
                    "prune_min_weight": {
                        "type": "number",
                        "description": "Minimum weight for edges to be kept (default: 0.3).",
                        "default": 0.3
                    },
                    "suggest_threshold": {
                        "type": "number",
                        "description": "Minimum similarity threshold for relation suggestions (default: 0.75).",
                        "default": 0.75
                    },
                    "suggest_max": {
                        "type": "integer",
                        "description": "Maximum number of relation suggestions (default: 10).",
                        "default": 10
                    },
                    "refine_similarity_threshold": {
                        "type": "number",
                        "description": "Minimum similarity threshold for summary refinement (default: 0.75). Notes with similar summaries will be made more specific.",
                        "default": 0.75
                    },
                    "refine_max": {
                        "type": "integer",
                        "description": "Maximum number of summaries to refine per run (default: 10).",
                        "default": 10
                    },
                    "auto_add_suggestions": {
                        "type": "boolean",
                        "description": "If true, automatically adds suggested relations to the graph instead of just suggesting them (default: false).",
                        "default": False
                    },
                    "ignore_flags": {
                        "type": "boolean",
                        "description": "If true, ignores validation flags and forces re-validation of all notes (summary, keywords, tags, edges). Useful for immediate corrections without waiting for flag expiration (default: false).",
                        "default": False
                    }
                }
            }
        ),
        Tool(
            name="research_and_store",
            description="Performs deep web research on a query and stores the findings as atomic notes. Uses Google Search API (if configured) or DuckDuckGo for web search, Jina Reader (local Docker or cloud) for web content extraction, and Unstructured for PDF extraction. Automatically detects PDF URLs and uses appropriate extraction method. Results are stored as atomic notes in the memory system with automatic linking and evolution.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The research query to search for on the web."
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional context about why this research is needed.",
                        "default": "Manual research request"
                    },
                    "max_sources": {
                        "type": "integer",
                        "description": "Maximum number of sources to extract and create notes from (default: 1). Lower values = fewer notes, higher values = more comprehensive research.",
                        "default": 1,
                        "minimum": 1,
                        "maximum": 20
                    }
                },
                "required": ["query"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> Sequence[TextContent]:
    """Executes a tool."""
    
    # Validate parameters for all tools
    from .utils.validation import validate_tool_parameters
    is_valid, validation_errors = validate_tool_parameters(name, arguments)
    if not is_valid:
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": "Parameter validation failed",
                "validation_errors": validation_errors
            }, indent=2)
        )]
    
    if name == "create_atomic_note":
        content = arguments.get("content", "")
        source = arguments.get("source", "user_input")
        
        if not content:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "content is required"}, indent=2)
            )]
        
        try:
            note_input = NoteInput(content=content, source=source)
            note_id = await controller.create_note(note_input)
            
            result = {
                "status": "success",
                "note_id": note_id,
                "message": f"Note created with ID: {note_id}. Evolution started in background."
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)}, indent=2)
            )]
    
    elif name == "retrieve_memories":
        query = arguments.get("query", "")
        max_results = int(arguments.get("max_results", 5))  # Normalize to int
        
        try:
            results = await controller.retrieve(query)
            
            # Limit results
            results = results[:max_results]
            
            output = []
            for res in results:
                context_str = ", ".join([f"[{rn.id}] {rn.contextual_summary}" for rn in res.related_notes])
                output.append({
                    "id": res.note.id,
                    "content": res.note.content,
                    "summary": res.note.contextual_summary,
                    "keywords": res.note.keywords,
                    "tags": res.note.tags,
                    "type": res.note.type,  # Node classification
                    "relevance_score": float(res.score),  # Combined similarity × priority
                    "connected_memories": len(res.related_notes),
                    "connected_context": context_str
                })
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "query": query,
                    "results_count": len(output),
                    "results": output
                }, indent=2, ensure_ascii=False)
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)}, indent=2)
            )]
    
    elif name == "get_memory_stats":
        try:
            graph = controller.storage.graph.graph
            stats = {
                "status": "success",
                "graph_nodes": graph.number_of_nodes(),
                "graph_edges": graph.number_of_edges(),
                "memory_count": graph.number_of_nodes(),
                "connection_count": graph.number_of_edges()
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(stats, indent=2)
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)}, indent=2)
            )]
    
    elif name == "add_file":
        file_path = arguments.get("file_path", "")
        file_content = arguments.get("file_content", "")
        chunk_size = int(arguments.get("chunk_size", 15000))  # Normalize to int
        
        # Check if file_path or file_content is provided
        if not file_path and not file_content:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "Either file_path or file_content is required"}, indent=2)
            )]
        
        try:
            # Read file if file_path is provided
            if file_path:
                path = Path(file_path)
                # Try absolute path first, then relative to current working directory
                if not path.is_absolute():
                    # Try relative to current working directory
                    cwd_path = Path.cwd() / path
                    if cwd_path.exists():
                        path = cwd_path
                    else:
                        # Try relative to server directory (where main.py is)
                        server_dir = Path(__file__).parent.parent.parent
                        server_path = server_dir / path
                        if server_path.exists():
                            path = server_path
                
                if not path.exists():
                    return [TextContent(
                        type="text",
                        text=json.dumps({
                            "error": f"File not found: {file_path}",
                            "hint": "Try using an absolute path or ensure the file is in the server directory"
                        }, indent=2)
                    )]
                
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                    source = f"file:{path.name}"
                except UnicodeDecodeError:
                    # Fallback for binary files
                    with open(path, 'rb') as f:
                        content_bytes = f.read()
                    file_content = content_bytes.decode('utf-8', errors='replace')
                    source = f"file:{path.name}"
            else:
                source = "file:direct_content"
            
            # Check size and chunk if necessary
            content_bytes = file_content.encode('utf-8')
            file_size = len(content_bytes)
            
            if file_size <= chunk_size:
                # File fits in one note
                note_input = NoteInput(content=file_content, source=source)
                note_id = await controller.create_note(note_input)
                
                result = {
                    "status": "success",
                    "note_id": note_id,
                    "file_size": file_size,
                    "chunks": 1,
                    "message": f"File stored as single note with ID: {note_id}. Evolution started in background."
                }
            else:
                # Chunking required
                chunks = []
                chunk_count = (file_size + chunk_size - 1) // chunk_size
                
                for i in range(chunk_count):
                    start = i * chunk_size
                    end = min(start + chunk_size, file_size)
                    chunk_content = content_bytes[start:end].decode('utf-8', errors='replace')
                    
                    # Add chunk info
                    chunk_header = f"[Chunk {i+1}/{chunk_count} from {source}]\n\n"
                    chunk_note_content = chunk_header + chunk_content
                    
                    note_input = NoteInput(
                        content=chunk_note_content,
                        source=f"{source}:chunk_{i+1}"
                    )
                    note_id = await controller.create_note(note_input)
                    chunks.append(note_id)
                
                result = {
                    "status": "success",
                    "file_size": file_size,
                    "chunks": chunk_count,
                    "note_ids": chunks,
                    "message": f"File split into {chunk_count} chunks. All notes created. Evolution started in background."
                }
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2, ensure_ascii=False)
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)}, indent=2)
            )]
    
    elif name == "reset_memory":
        try:
            success = await controller.reset_memory()
            
            if success:
                result = {
                    "status": "success",
                    "message": "Memory system reset successfully. All notes, edges, and embeddings have been deleted."
                }
            else:
                result = {
                    "status": "error",
                    "message": "Failed to reset memory system. Check logs for details."
                }
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)}, indent=2)
            )]
    
    elif name == "list_notes":
        notes = await controller.list_notes_data()
        return [TextContent(
            type="text",
            text=json.dumps({"notes": notes}, indent=2, ensure_ascii=False)
        )]
    
    elif name == "get_note":
        note_id = arguments.get("note_id", "")
        if not note_id:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "note_id is required"}, indent=2)
            )]
        note = await controller.get_note_data(note_id)
        if not note:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Note '{note_id}' not found"}, indent=2)
            )]
        return [TextContent(
            type="text",
            text=json.dumps({"note": note}, indent=2, ensure_ascii=False)
        )]
    
    elif name == "update_note":
        note_id = arguments.get("note_id", "")
        data = arguments.get("data", {})
        if not note_id:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "note_id is required"}, indent=2)
            )]
        if not isinstance(data, dict):
            return [TextContent(
                type="text",
                text=json.dumps({"error": "data must be an object"}, indent=2)
            )]
        result = await controller.update_note_metadata(note_id, data)
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2, ensure_ascii=False)
        )]
    
    elif name == "delete_atomic_note":
        note_id = arguments.get("note_id", "")
        if not note_id:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "note_id is required"}, indent=2)
            )]
        result = await controller.delete_note_data(note_id)
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    
    elif name == "list_relations":
        note_id = arguments.get("note_id")
        relations = await controller.list_relations_data(note_id)
        return [TextContent(
            type="text",
            text=json.dumps({"relations": relations}, indent=2, ensure_ascii=False)
        )]
    
    elif name == "add_relation":
        source = arguments.get("source_id", "")
        target = arguments.get("target_id", "")
        relation_type = arguments.get("relation_type", "relates_to")
        reasoning = arguments.get("reasoning", "Manual link")
        weight = float(arguments.get("weight", 1.0))  # Normalize to float
        
        # Additional validation: weight must be 0.0-1.0
        if weight < 0.0 or weight > 1.0:
            weight = max(0.0, min(1.0, weight))  # Clamp to valid range
        result = await controller.add_relation(source, target, relation_type, reasoning, weight)
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2, ensure_ascii=False)
        )]
    
    elif name == "remove_relation":
        source = arguments.get("source_id", "")
        target = arguments.get("target_id", "")
        if not source or not target:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "source_id and target_id are required"}, indent=2)
            )]
        result = await controller.remove_relation(source, target)
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    
    elif name == "get_graph":
        save_to_disk = arguments.get("save", False)
        
        # Get graph snapshot first to see what we have
        graph = await controller.get_graph_snapshot()
        node_count = len(graph.get("nodes", []))
        edge_count = len(graph.get("edges", []))
        
        # Log to file
        from datetime import datetime
        # Import settings explicitly to avoid scope issues
        from .config import settings as config_settings
        log_file = Path(config_settings.DATA_DIR) / "graph_save.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        def write_log(msg):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] {msg}\n")
            log_debug(msg)
        
        write_log(f"[INFO] [get_graph] Graph snapshot: {node_count} nodes, {edge_count} edges")
        write_log(f"[INFO] [get_graph] Memory graph: {controller.storage.graph.graph.number_of_nodes()} nodes, {controller.storage.graph.graph.number_of_edges()} edges")
        
        # If save is requested, save the graph to disk
        if save_to_disk:
            loop = asyncio.get_running_loop()
            write_log(f"[SAVE] [get_graph] Saving graph to disk...")
            await loop.run_in_executor(None, controller.storage.graph.save_snapshot)
            write_log(f"[SAVE] [get_graph] Save completed")
            
            # Verify after save
            # Check if using GraphML (RustworkX) or JSON (NetworkX)
            graphml_path = config_settings.GRAPH_PATH.with_suffix(".graphml")
            if graphml_path.exists():
                # RustworkX: GraphML format - verify using graph methods
                saved_nodes = controller.storage.graph.number_of_nodes()
                saved_links = controller.storage.graph.number_of_edges()
                log_debug(f"[OK] [get_graph] Verified GraphML file: {saved_nodes} nodes, {saved_links} edges")
                if saved_nodes == 0 and node_count > 0:
                    write_log(f"[ERROR] [get_graph] ERROR: Graph had {node_count} nodes but saved file has 0 nodes!")
            elif config_settings.GRAPH_PATH.exists():
                # NetworkX: JSON format
                with open(config_settings.GRAPH_PATH, 'r', encoding='utf-8') as f:
                    saved_data = json.load(f)
                    saved_nodes = len(saved_data.get("nodes", []))
                    saved_links = len(saved_data.get("links", []))
                    log_debug(f"[OK] [get_graph] Verified saved file: {saved_nodes} nodes, {saved_links} links")
                    if saved_nodes == 0 and node_count > 0:
                        write_log(f"[ERROR] [get_graph] ERROR: Graph had {node_count} nodes but saved file has 0 nodes!")
        
        result = {
            "nodes": graph.get("nodes", []),
            "edges": graph.get("edges", []),
            "saved_to_disk": save_to_disk,
            "memory_nodes": controller.storage.graph.graph.number_of_nodes(),
            "memory_edges": controller.storage.graph.graph.number_of_edges()
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2, ensure_ascii=False)
        )]
    
    elif name == "run_memory_enzymes":
        try:
            prune_max_age = arguments.get("prune_max_age_days", 90)
            prune_min_weight = arguments.get("prune_min_weight", 0.3)
            suggest_threshold = arguments.get("suggest_threshold", 0.75)
            suggest_max = arguments.get("suggest_max", 10)
            auto_add_suggestions = arguments.get("auto_add_suggestions", False)  # Neu: Auto-Add Relations
            
            # Run enzymes synchronously (in thread)
            loop = asyncio.get_running_loop()
            
            refine_similarity = arguments.get("refine_similarity_threshold", 0.75)  # Niedrigerer Default für bessere Erkennung
            refine_max = arguments.get("refine_max", 10)
            ignore_flags = arguments.get("ignore_flags", False)  # Neu: Flags ignorieren
            
            def _run_enzymes():
                return run_memory_enzymes(
                    controller.storage.graph,
                    controller.llm,
                    prune_config={
                        "max_age_days": prune_max_age,
                        "min_weight": prune_min_weight
                    },
                    suggest_config={
                        "threshold": suggest_threshold,
                        "max_suggestions": suggest_max
                    },
                    refine_config={
                        "similarity_threshold": refine_similarity,
                        "max_refinements": refine_max
                    },
                    auto_add_suggestions=auto_add_suggestions,
                    ignore_flags=ignore_flags
                )
            
            results = await loop.run_in_executor(None, _run_enzymes)
            
            # Log manual enzyme run (use the same log_event from priority module)
            from .utils.priority import log_event
            log_event("ENZYME_MANUAL_RUN", {
                "results": results,
                "prune_max_age_days": prune_max_age,
                "prune_min_weight": prune_min_weight,
                "suggest_threshold": suggest_threshold,
                "suggest_max": suggest_max,
                "refine_similarity_threshold": refine_similarity,
                "refine_max": refine_max
            })
            
            # Save graph after enzymes
            await loop.run_in_executor(None, controller.storage.graph.save_snapshot)
            
            summaries_refined = results.get("summaries_refined", 0)
            suggestions = results.get("suggestions", [])
            relations_added = results.get("relations_auto_added", 0)
            
            duplicates_merged = results.get("duplicates_merged", 0)
            self_loops_removed = results.get("self_loops_removed", 0)
            isolated_nodes = results.get("isolated_nodes_found", 0)
            isolated_linked = results.get("isolated_nodes_linked", 0)
            low_quality_removed = results.get("low_quality_notes_removed", 0)
            notes_validated = results.get("notes_validated", 0)
            notes_corrected = results.get("notes_corrected", 0)
            
            message_parts = [
                f"{results['pruned_count']} links pruned",
                f"{results.get('zombie_nodes_removed', 0)} zombie nodes removed"
            ]
            
            if low_quality_removed > 0:
                message_parts.append(f"{low_quality_removed} low-quality notes removed")
            
            if self_loops_removed > 0:
                message_parts.append(f"{self_loops_removed} self-loops removed")
            
            if duplicates_merged > 0:
                message_parts.append(f"{duplicates_merged} duplicates merged")
            
            if notes_validated > 0:
                message_parts.append(f"{notes_validated} notes validated")
            
            if notes_corrected > 0:
                message_parts.append(f"{notes_corrected} notes corrected")
            
            quality_scores_calculated = results.get("quality_scores_calculated", 0)
            low_quality_count = len(results.get("low_quality_notes", []))
            
            if quality_scores_calculated > 0:
                message_parts.append(f"{quality_scores_calculated} quality scores calculated")
            if low_quality_count > 0:
                message_parts.append(f"{low_quality_count} low-quality notes found")
            
            keywords_normalized = results.get("keywords_normalized", 0)
            keywords_removed = results.get("keywords_removed", 0)
            keywords_corrected = results.get("keywords_corrected", 0)
            
            if keywords_normalized > 0:
                message_parts.append(f"{keywords_normalized} keywords normalized")
            if keywords_removed > 0:
                message_parts.append(f"{keywords_removed} keywords removed")
            if keywords_corrected > 0:
                message_parts.append(f"{keywords_corrected} keywords corrected")
            
            if isolated_nodes > 0:
                message_parts.append(f"{isolated_nodes} isolated nodes found")
            
            if isolated_linked > 0:
                message_parts.append(f"{isolated_linked} isolated nodes auto-linked")
            
            edges_removed = results.get("edges_removed", 0)
            reasonings_added = results.get("reasonings_added", 0)
            types_standardized = results.get("types_standardized", 0)
            
            if edges_removed > 0:
                message_parts.append(f"{edges_removed} weak edges removed")
            if reasonings_added > 0:
                message_parts.append(f"{reasonings_added} reasonings added")
            if types_standardized > 0:
                message_parts.append(f"{types_standardized} relation types standardized")
            
            message_parts.extend([
                f"{summaries_refined} summaries refined",
                f"{results['suggestions_count']} relations suggested"
            ])
            
            if relations_added > 0:
                message_parts.append(f"{relations_added} relations auto-added")
            
            message_parts.append(f"{results['digested_count']} nodes digested")
            
            # Neue Funktionen
            types_validated = results.get("types_validated", 0)
            types_corrected = results.get("types_corrected", 0)
            if types_validated > 0:
                message_parts.append(f"{types_validated} types validated")
            if types_corrected > 0:
                message_parts.append(f"{types_corrected} types corrected")
            
            notes_archived = results.get("notes_archived", 0)
            notes_deleted = results.get("notes_deleted", 0)
            if notes_archived > 0:
                message_parts.append(f"{notes_archived} notes archived")
            if notes_deleted > 0:
                message_parts.append(f"{notes_deleted} notes deleted")
            
            graph_health_score = results.get("graph_health_score")
            graph_health_level = results.get("graph_health_level", "unknown")
            if graph_health_score is not None:
                message_parts.append(f"graph health: {graph_health_score:.2f} ({graph_health_level})")
            
            dead_end_count = results.get("dead_end_nodes_found", 0)
            if dead_end_count > 0:
                message_parts.append(f"{dead_end_count} dead-end nodes found")
            
            isolated_nodes_list = results.get("isolated_nodes", [])
            
            response_data = {
                "status": "success",
                "results": results,
                "message": f"Enzymes completed: {', '.join(message_parts)}.",
                "suggested_relations": suggestions  # Vorgeschlagene Relations für User-Review
            }
            
            if isolated_nodes_list:
                response_data["isolated_nodes"] = isolated_nodes_list  # Isolierte Nodes für User-Review
            
            low_quality_notes = results.get("low_quality_notes", [])
            if low_quality_notes:
                response_data["low_quality_notes"] = low_quality_notes  # Notes mit niedrigem Quality-Score
            
            # Neue Funktionen: Graph Health & Dead-Ends
            graph_health_details = results.get("graph_health_details")
            if graph_health_details:
                response_data["graph_health_details"] = graph_health_details
            
            dead_end_nodes = results.get("dead_end_nodes", [])
            if dead_end_nodes:
                response_data["dead_end_nodes"] = dead_end_nodes
            
            # Add validation warnings if any
            validation_warnings = results.get("validation_warnings", [])
            if validation_warnings:
                response_data["validation_warnings"] = validation_warnings
                response_data["message"] += f" Note: {len(validation_warnings)} parameter(s) were corrected to default values."
            
            return [TextContent(
                type="text",
                text=json.dumps(response_data, indent=2, ensure_ascii=False)
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)}, indent=2)
        )]
    
    elif name == "research_and_store":
        query = arguments.get("query", "")
        context = arguments.get("context", "Manual research request")
        max_sources = int(arguments.get("max_sources", 1))  # Normalize to int (already validated)
        
        try:
            from .utils.researcher import ResearcherAgent
            from .utils.priority import log_event
            
            log_debug(f"[RESEARCHER] Starting research for: {query} (max_sources: {max_sources})")
            log_event("RESEARCHER_MANUAL_RUN_START", {
                "query": query,
                "context": context,
                "max_sources": max_sources
            })
            
            # Initialize researcher with configurable max_sources
            # Note: MCP tool callback is not available in this context (would require MCP client)
            # Researcher will use HTTP-based fallbacks (Google Search API, DuckDuckGo, Jina Reader HTTP)
            researcher = ResearcherAgent(llm_service=controller.llm, max_sources=max_sources, mcp_tool_callback=None)
            
            # Researcher Agent uses HTTP-based tools (researcher_tools.py) by default:
            # Strategy: HTTP-based tools (Google Search API, DuckDuckGo HTTP, Jina Reader HTTP)
            # If MCP tools are available via callback, they will be tried first, then HTTP fallbacks.
            
            # Perform research (uses HTTP-based tools directly)
            research_notes = await researcher.research(query=query, context=context)
            
            if not research_notes:
                # Provide helpful error message about why research failed
                error_details = []
                from ..config import settings
                
                if not settings.GOOGLE_SEARCH_ENABLED or not settings.GOOGLE_API_KEY:
                    error_details.append("Google Search API not configured (check GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID)")
                
                if not settings.JINA_READER_ENABLED:
                    error_details.append("Jina Reader not enabled (check JINA_READER_ENABLED)")
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "info",
                        "message": "Research tool executed but no notes created. Possible reasons: no search results found, content extraction failed, or all content was filtered as irrelevant.",
                        "notes_created": 0,
                        "notes_stored": 0,
                        "debug_info": {
                            "query": query,
                            "max_sources": max_sources,
                            "possible_issues": error_details if error_details else [
                                "No search results found for query",
                                "Content extraction failed (check Jina Reader/Unstructured)",
                                "All extracted content was filtered as irrelevant",
                                "Check server console output for detailed error messages"
                            ]
                        },
                        "tip": "Check server console output for detailed debug messages (prefixed with [RESEARCHER])"
                    }, indent=2)
                )]
            
            # Store research notes
            notes_stored = []
            notes_failed = []
            
            for note in research_notes:
                try:
                    # Pass full metadata from ResearcherAgent (avoids duplicate LLM extraction)
                    note_input = NoteInput(
                        content=note.content,
                        source="researcher_agent",
                        contextual_summary=note.contextual_summary,
                        keywords=note.keywords,
                        tags=note.tags,
                        type=note.type,
                        metadata=note.metadata
                    )
                    note_id = await controller.create_note(note_input)
                    notes_stored.append({
                        "id": note_id,
                        "summary": note.contextual_summary,
                        "type": note.type,
                        "source_url": note.metadata.get("source_url", "N/A")
                    })
                except Exception as e:
                    notes_failed.append({
                        "error": str(e),
                        "note_id": note.id[:8] if note else "unknown"
                    })
            
            # Log event
            log_event("RESEARCHER_MANUAL_RUN", {
                "query": query,
                "context": context,
                "notes_created": len(research_notes),
                "notes_stored": len(notes_stored),
                "notes_failed": len(notes_failed)
            })
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "message": f"Research completed: {len(notes_stored)} notes stored",
                    "query": query,
                    "notes_created": len(research_notes),
                    "notes_stored": len(notes_stored),
                    "notes_failed": len(notes_failed),
                    "stored_notes": notes_stored,
                    "failed_notes": notes_failed if notes_failed else None
                }, indent=2, ensure_ascii=False)
            )]
        except Exception as e:
            from .utils.priority import log_event
            log_event("RESEARCHER_ERROR", {
                "query": query,
                "error": str(e)
            })
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)}, indent=2)
            )]
    
    else:
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"Unknown tool: {name}"}, indent=2)
        )]

async def main():
    """Main function for the MCP Server."""
    # Initial save: Save graph to disk on startup so visualizer can access it
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, controller.storage.graph.save_snapshot)
    log_debug("[SAVE] Initial graph snapshot saved to disk")
    
    # Starte automatischen Enzyme-Scheduler (alle 1 Stunde)
    # Auto-Save alle 5 Minuten, damit Visualizer die Daten sehen kann
    controller.start_enzyme_scheduler(interval_hours=1.0, auto_save_interval_minutes=5.0)
    
    # Starte HTTP-Server parallel (wenn enabled) - nutzt GLEICHE controller-Instanz
    http_task = None
    if settings.TCP_SERVER_ENABLED:
        from aiohttp import web
        
        async def get_graph_handler(request):
            """HTTP Handler für get_graph"""
            graph_data = await controller.get_graph_snapshot()
            return web.json_response(graph_data)
        
        async def run_http_server():
            app = web.Application()
            app.router.add_get('/get_graph', get_graph_handler)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, settings.TCP_SERVER_HOST, settings.TCP_SERVER_PORT)
            await site.start()
            log_debug(f"[HTTP] HTTP-Server gestartet auf http://{settings.TCP_SERVER_HOST}:{settings.TCP_SERVER_PORT}/get_graph")
            # Lauf ewig
            await asyncio.Event().wait()
        
        http_task = asyncio.create_task(run_http_server())
    
    # Haupt-Server: stdio für Cursor/IDE
    async with stdio_server() as (read_stream, write_stream):
        try:
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )
        finally:
            # Stoppe HTTP-Server wenn aktiv
            if http_task:
                http_task.cancel()
                try:
                    await http_task
                except asyncio.CancelledError:
                    pass
                log_debug("[STOP] HTTP-Server gestoppt")
            
            # Final save: Save graph to disk on shutdown
            await loop.run_in_executor(None, controller.storage.graph.save_snapshot)
            log_debug("[SAVE] Final graph snapshot saved to disk")
            # Stoppe Scheduler beim Shutdown
            controller.stop_enzyme_scheduler()

if __name__ == "__main__":
    asyncio.run(main())
