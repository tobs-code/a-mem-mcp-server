"""
Storage Engine: GraphStore, VectorStore, StorageManager

Implements Cross-Platform Locking, safe loading and separate saving from adding.
"""

import json
import shutil
import networkx as nx
import chromadb
from typing import List, Dict, Tuple, Optional
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from ..config import settings
from ..models.note import AtomicNote, NoteRelation

# Import RustworkXGraphStore if available
try:
    from .rustworkx_store import RustworkXGraphStore, RUSTWORKX_AVAILABLE
except ImportError:
    RustworkXGraphStore = None
    RUSTWORKX_AVAILABLE = False

# Import SafeGraphStore wrapper
try:
    from .safe_graph_wrapper import SafeGraphStore
    SAFE_WRAPPER_AVAILABLE = True
except ImportError:
    SafeGraphStore = None
    SAFE_WRAPPER_AVAILABLE = False

# Import FalkorDBGraphStore if available
try:
    from .falkordb_store import FalkorDBGraphStore, FALKORDB_AVAILABLE
except ImportError:
    FalkorDBGraphStore = None
    FALKORDB_AVAILABLE = False

# Import Windows adapter if on Windows
FALKORDB_WINDOWS_AVAILABLE = False
FalkorDBGraphStoreWindows = None
if sys.platform == 'win32':
    try:
        from .falkordb_store_windows import FalkorDBGraphStoreWindows
        FALKORDB_WINDOWS_AVAILABLE = True
    except ImportError:
        pass

# Log file path
LOG_FILE = settings.DATA_DIR / "graph_save.log"

def _write_log(message: str):
    """Writes a log message to file and stderr (not stdout, to avoid breaking MCP JSON-RPC)."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"
    
    # Write to file
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    except Exception as e:
        print(f"[WARNING] Failed to write log: {e}", file=sys.stderr)
    
    # Print to stderr (not stdout) to avoid breaking MCP JSON-RPC
    print(message, file=sys.stderr)

# --- Cross-Platform Locking ---
try:
    # 1. Versuch POSIX fcntl (Linux/macOS)
    import fcntl
    def lock_file(f): fcntl.flock(f, fcntl.LOCK_EX)
    def unlock_file(f): fcntl.flock(f, fcntl.LOCK_UN)
except ImportError:
    try:
        # 2. Versuch portalocker (Windows/Cross-Platform)
        import portalocker
        def lock_file(f): portalocker.lock(f, portalocker.LOCK_EX)
        def unlock_file(f): portalocker.unlock(f)
    except ImportError:
        # 3. Letzter Fallback (Kein echtes Locking)
        def lock_file(f): 
            print("WARNING: No file locking available (Install 'portalocker' for Windows support).", file=sys.stderr)
            pass
        def unlock_file(f): pass

class GraphStore:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.load()

    @contextmanager
    def _file_lock(self):
        """Cross-platform File Lock."""
        with open(settings.LOCK_FILE, 'w') as lock_f:
            try:
                lock_file(lock_f)
                yield
            finally:
                unlock_file(lock_f)

    def load(self):
        """Loads the graph safely. Prevents data loss on corrupted JSON."""
        if settings.GRAPH_PATH.exists():
            try:
                with open(settings.GRAPH_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.graph = nx.node_link_graph(data)
            except json.JSONDecodeError:
                timestamp = os.urandom(4).hex()
                backup_path = settings.GRAPH_PATH.with_suffix(f".bak.{timestamp}")
                print(f"CRITICAL: Graph JSON corrupted. Backing up to {backup_path}")
                shutil.copy(settings.GRAPH_PATH, backup_path)
                raise RuntimeError("Graph database is corrupted. Check backup.")
            except Exception as e:
                print(f"Error loading graph: {e}", file=sys.stderr)
                # Only abort on read problems (Permissions etc), 
                # on 'Not Found' a new one is created.
                if not os.access(settings.GRAPH_PATH, os.R_OK):
                     raise
                self.graph = nx.DiGraph()
        else:
            self.graph = nx.DiGraph()

    def save_snapshot(self):
        """Saves the current state to disk."""
        node_count = self.graph.number_of_nodes()
        edge_count = self.graph.number_of_edges()
        _write_log(f"[SAVE] [save_snapshot] Saving graph: {node_count} nodes, {edge_count} edges")
        
        with self._file_lock():
            data = nx.node_link_data(self.graph)
            nodes_in_data = len(data.get("nodes", []))
            links_in_data = len(data.get("links", []))
            _write_log(f"[SAVE] [save_snapshot] Data to save: {nodes_in_data} nodes, {links_in_data} links")
            
            # Atomic write pattern: Write to temp, then rename
            temp_path = settings.GRAPH_PATH.with_suffix(".tmp")
            try:
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                _write_log(f"[SAVE] [save_snapshot] Temp file written: {temp_path}")
            except Exception as e:
                _write_log(f"[ERROR] [save_snapshot] Error writing temp file: {e}")
                raise
            
            # Verify temp file
            if temp_path.exists():
                temp_size = temp_path.stat().st_size
                _write_log(f"[SAVE] [save_snapshot] Temp file size: {temp_size} bytes")
            else:
                _write_log(f"[ERROR] [save_snapshot] Temp file does not exist after write!")
            
            try:
                os.replace(temp_path, settings.GRAPH_PATH)
                _write_log(f"[SAVE] [save_snapshot] File replaced: {settings.GRAPH_PATH}")
            except Exception as e:
                _write_log(f"[ERROR] [save_snapshot] Error replacing file: {e}")
                raise
            
            # Verify final file
            if settings.GRAPH_PATH.exists():
                final_size = settings.GRAPH_PATH.stat().st_size
                _write_log(f"[SAVE] [save_snapshot] Final file size: {final_size} bytes")
                # Quick verification
                try:
                    with open(settings.GRAPH_PATH, 'r', encoding='utf-8') as f:
                        verify_data = json.load(f)
                        verify_nodes = len(verify_data.get("nodes", []))
                        verify_links = len(verify_data.get("links", []))
                        _write_log(f"[SAVE] [save_snapshot] Verified: {verify_nodes} nodes, {verify_links} links in saved file")
                        
                        if verify_nodes == 0 and node_count > 0:
                            _write_log(f"[WARNING] [save_snapshot] WARNING: Graph had {node_count} nodes but saved file has 0 nodes!")
                except Exception as e:
                    _write_log(f"[WARNING] [save_snapshot] Verification failed: {e}")
            else:
                _write_log(f"[ERROR] [save_snapshot] Final file does not exist after replace!")

    def add_node(self, note: AtomicNote):
        """Adds node (In-Memory). Saving must be triggered separately."""
        self.graph.add_node(note.id, **note.model_dump(mode='json'))

    def add_edge(self, relation: NoteRelation):
        """Adds edge (In-Memory)."""
        self.graph.add_edge(
            relation.source_id, 
            relation.target_id, 
            type=relation.relation_type,
            reasoning=relation.reasoning,
            weight=relation.weight,
            created_at=relation.created_at.isoformat()  # Store as ISO string for JSON serialization
        )

    def update_node(self, note: AtomicNote):
        """Updates an existing node in the graph (Memory Evolution)."""
        if note.id not in self.graph:
            # If node doesn't exist, add it
            self.add_node(note)
        else:
            # Update node attributes
            self.graph.nodes[note.id].update(note.model_dump(mode='json'))

    def get_neighbors(self, node_id: str) -> List[Dict]:
        if node_id not in self.graph:
            return []
        neighbors = list(self.graph.successors(node_id)) + list(self.graph.predecessors(node_id))
        results = []
        for n_id in set(neighbors):
            if n_id in self.graph.nodes:
                results.append(self.graph.nodes[n_id])
        return results
    
    def remove_node(self, node_id: str):
        """Removes a node and all associated edges (In-Memory)."""
        if node_id in self.graph:
            # NetworkX automatically removes all edges when deleting a node
            self.graph.remove_node(node_id)
    
    def reset(self):
        """Resets the graph completely (In-Memory + file)."""
        # Delete graph file if it exists
        if settings.GRAPH_PATH.exists():
            settings.GRAPH_PATH.unlink()
        
        # Create new empty graph
        self.graph = nx.DiGraph()
        
        # Save empty graph
        self.save_snapshot()

class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(settings.CHROMA_DIR))
        # Create collection with explicit cosine similarity metric for better semantic search
        self.collection = self.client.get_or_create_collection(
            name="memories",
            metadata={"hnsw:space": "cosine"}  # Explicit cosine similarity
        )
        # Erwartete Embedding-Dimension basierend auf Provider
        self._expected_dimension = self._get_expected_dimension()
        self._validate_dimension()
    
    def _get_expected_dimension(self) -> int:
        """Determines the expected embedding dimension based on the configured model."""
        from ..config import settings
        if settings.LLM_PROVIDER == "openrouter":
            # OpenRouter Embedding Models - bekannte Dimensionen
            model = settings.OPENROUTER_EMBEDDING_MODEL.lower()
            if "text-embedding-3-small" in model:
                return 1536
            elif "text-embedding-3-large" in model:
                return 3072
            elif "qwen3-embedding-8b" in model or "qwen/qwen3-embedding-8b" in model:
                return 4096
            elif "text-embedding-ada-002" in model:
                return 1536
            else:
                # Fallback: Versuche aus Model-Name zu extrahieren oder Standard
                print(f"Warning: Unknown OpenRouter embedding model '{model}'. Assuming 1536 dimensions.", file=sys.stderr)
                return 1536
        else:
            # Ollama Embedding Models
            model = settings.OLLAMA_EMBEDDING_MODEL.lower()
            if "nomic-embed-text" in model:
                return 768
            elif "all-minilm" in model:
                return 384
            else:
                # Fallback for unknown Ollama models
                print(f"Warning: Unknown Ollama embedding model '{model}'. Assuming 768 dimensions.", file=sys.stderr)
                return 768
    
    def _validate_dimension(self):
        """Validates whether the collection is compatible with the expected dimension."""
        try:
            # Try to create a test embedding (only check dimension)
            test_embedding = [0.0] * self._expected_dimension
            # Check if collection already has items
            if self.collection.count() > 0:
                # Get an existing item to check the current dimension
                sample = self.collection.get(limit=1)
                if sample['ids']:
                    # ChromaDB stores embeddings internally, we can't directly query the dimension
                    # But we can try to add a new item with the expected dimension
                    # and see if it fails
                    pass
        except Exception as e:
            print(f"Warning: Dimension validation error: {e}", file=sys.stderr)
    
    def _check_dimension_compatibility(self, embedding: List[float]) -> bool:
        """Checks if the embedding matches the expected dimension."""
        actual_dim = len(embedding)
        if actual_dim != self._expected_dimension:
            from ..config import settings
            print(f"[WARNING] CRITICAL: Embedding dimension mismatch!", file=sys.stderr)
            print(f"   Expected: {self._expected_dimension} (from {settings.LLM_PROVIDER}/{settings.EMBEDDING_MODEL})", file=sys.stderr)
            print(f"   Actual: {actual_dim}", file=sys.stderr)
            print(f"   This will cause ChromaDB errors!", file=sys.stderr)
            print(f"   Solution: Delete 'data/chroma' directory and restart, or use consistent embedding models.", file=sys.stderr)
            return False
        return True

    def add(self, note: AtomicNote, embedding: List[float]):
        # Dimension validation
        if not self._check_dimension_compatibility(embedding):
            raise ValueError(
                f"Embedding dimension mismatch. Expected {self._expected_dimension}, got {len(embedding)}. "
                f"Delete 'data/chroma' directory to reset or use consistent embedding models."
            )
        
        # ChromaDB handles internal locks mostly itself, but we use it sync.
        # In the Logic Layer this is offloaded to a thread.
        self.collection.add(
            ids=[note.id],
            embeddings=[embedding],
            documents=[note.content],
            metadatas=[{
                "timestamp": note.created_at.isoformat(),
                "summary": note.contextual_summary
            }]
        )

    def query(self, embedding: List[float], k: int = 5) -> Tuple[List[str], List[float]]:
        # Dimension-Validierung
        if not self._check_dimension_compatibility(embedding):
            raise ValueError(
                f"Query embedding dimension mismatch. Expected {self._expected_dimension}, got {len(embedding)}. "
                f"Delete 'data/chroma' directory to reset or use consistent embedding models."
            )
        
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=k
        )
        if not results['ids']:
            return [], []
        return results['ids'][0], results['distances'][0]

    def update(self, note_id: str, note: AtomicNote, embedding: List[float]):
        """Updates an existing note in ChromaDB (Memory Evolution)."""
        # Dimension validation
        if not self._check_dimension_compatibility(embedding):
            raise ValueError(
                f"Update embedding dimension mismatch. Expected {self._expected_dimension}, got {len(embedding)}. "
                f"Delete 'data/chroma' directory to reset or use consistent embedding models."
            )
        
        # ChromaDB supports update() - if not available, delete + add
        try:
            # Try update (if ChromaDB version supports it)
            self.collection.update(
                ids=[note_id],
                embeddings=[embedding],
                documents=[note.content],
                metadatas=[{
                    "timestamp": note.created_at.isoformat(),
                    "summary": note.contextual_summary
                }]
            )
        except AttributeError:
            # Fallback: Delete + Add (for older ChromaDB versions)
            self.collection.delete(ids=[note_id])
            self.add(note, embedding)
    
    def delete(self, note_id: str):
        """Deletes a note from ChromaDB."""
        try:
            self.collection.delete(ids=[note_id])
        except Exception as e:
            print(f"Warning: Error deleting note {note_id} from vector store: {e}", file=sys.stderr)
    
    def reset(self):
        """Resets the ChromaDB collection completely."""
        try:
            # Delete collection
            self.client.delete_collection(name="memories")
            # Create new empty collection
            self.collection = self.client.get_or_create_collection(name="memories")
        except Exception as e:
            print(f"Warning: Error resetting vector store: {e}", file=sys.stderr)
            # Fallback: Try to recreate collection
            try:
                self.collection = self.client.get_or_create_collection(name="memories")
            except Exception as e2:
                print(f"Critical: Could not recreate collection: {e2}", file=sys.stderr)
                raise

def create_graph_store():
    """
    Factory function to create the appropriate GraphStore implementation.
    
    Returns:
        GraphStore instance (NetworkX, RustworkX, FalkorDBLite, or FalkorDB-Windows based on config and platform)
    """
    backend = settings.GRAPH_BACKEND.lower()
    
    if backend == "rustworkx":
        # RustworkX (fastest option, Windows-compatible)
        if RUSTWORKX_AVAILABLE and RustworkXGraphStore is not None:
            try:
                _write_log("[GraphStore] Using RustworkX (GraphML persistence)")
                rustworkx_store = RustworkXGraphStore()
                # Wrap with SafeGraphStore for edge case handling
                if SAFE_WRAPPER_AVAILABLE and SafeGraphStore is not None:
                    return SafeGraphStore(rustworkx_store)
                else:
                    return rustworkx_store
            except Exception as e:
                _write_log(f"[GraphStore] RustworkX initialization failed: {e}")
                _write_log("[GraphStore] Falling back to NetworkX")
                return GraphStore()
        else:
            _write_log(
                "Warning: RustworkX not available. Install with: pip install rustworkx",
                file=sys.stderr
            )
            _write_log("[GraphStore] Falling back to NetworkX")
            return GraphStore()
    
    elif backend == "falkordb":
        # On Windows, try Windows adapter first (uses external Redis)
        if sys.platform == 'win32' and FALKORDB_WINDOWS_AVAILABLE and FalkorDBGraphStoreWindows is not None:
            try:
                _write_log("[GraphStore] Using FalkorDB Windows adapter (external Redis)")
                return FalkorDBGraphStoreWindows()
            except Exception as e:
                _write_log(f"[GraphStore] Windows adapter failed: {e}")
                _write_log("[GraphStore] Make sure Redis is running with FalkorDB module!")
                # Fall through to try FalkorDBLite or NetworkX
        
        # Try FalkorDBLite (Linux/macOS or if Windows adapter failed)
        if FALKORDB_AVAILABLE and FalkorDBGraphStore is not None:
            try:
                _write_log("[GraphStore] Using FalkorDBLite (embedded Redis)")
                return FalkorDBGraphStore()
            except Exception as e:
                _write_log(f"[GraphStore] FalkorDBLite initialization failed: {e}")
                # Fall through to NetworkX
        
        # Fallback to NetworkX
        _write_log(
            "Warning: FalkorDB not available. Falling back to NetworkX.",
            file=sys.stderr
        )
        if sys.platform == 'win32':
            _write_log(
                "On Windows, install: pip install falkordb redis (and run Redis with FalkorDB module)",
                file=sys.stderr
            )
        else:
            _write_log(
                "Install with: pip install falkordblite",
                file=sys.stderr
            )
        return GraphStore()
    else:
        # Default: NetworkX
        return GraphStore()


class StorageManager:
    """Facade for vector and graph storage."""
    def __init__(self):
        self.vector = VectorStore()
        self.graph = create_graph_store()
    
    def get_note(self, note_id: str) -> Optional[AtomicNote]:
        # Use get_node() if available (FalkorDB), otherwise fallback to direct access
        if hasattr(self.graph, 'get_node'):
            node_data = self.graph.get_node(note_id)
        else:
            # NetworkX fallback
            node_data = self.graph.graph.nodes.get(note_id) if hasattr(self.graph, 'graph') else None
        
        if node_data:
            # Ensure Pydantic validation, in case fields are missing
            try:
                return AtomicNote(**node_data)
            except Exception as e:
                print(f"Warning: Node {note_id} corrupted: {e}", file=sys.stderr)
                return None
        return None
    
    def delete_note(self, note_id: str) -> bool:
        """Deletes a note from Graph and Vector Store."""
        # Check if note exists
        if hasattr(self.graph, 'has_node'):
            exists = self.graph.has_node(note_id)
        else:
            # NetworkX fallback
            exists = note_id in self.graph.graph if hasattr(self.graph, 'graph') else False
        
        if not exists:
            return False
        
        try:
            # Delete from graph (automatically removes all edges)
            self.graph.remove_node(note_id)
            # Delete from vector store
            self.vector.delete(note_id)
            return True
        except Exception as e:
            print(f"Error deleting note {note_id}: {e}", file=sys.stderr)
            return False
    
    def reset(self):
        """Resets Graph and Vector Store completely."""
        # Reset graph
        self.graph.reset()
        # Reset vector store
        self.vector.reset()

