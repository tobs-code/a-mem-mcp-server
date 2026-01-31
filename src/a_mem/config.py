"""
Configuration for A-MEM System
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Lade .env Datei
load_dotenv()

class Config:
    # Paths - Use absolute path based on file location, not cwd
    # This ensures the paths work regardless of where the script is called from
    _config_file = Path(__file__).resolve()
    BASE_DIR = _config_file.parent.parent.parent  # Go up from src/a_mem/config.py to project root
    DATA_DIR = BASE_DIR / "data"
    CHROMA_DIR = DATA_DIR / "chroma"
    GRAPH_DIR = DATA_DIR / "graph"
    GRAPH_PATH = GRAPH_DIR / "knowledge_graph.json"
    
    # LLM Provider Selection
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()  # "ollama" oder "openrouter"
    
    # Ollama Settings
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen3:4b")
    OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest")
    
    # OpenRouter Settings
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    OPENROUTER_LLM_MODEL = os.getenv("OPENROUTER_LLM_MODEL", "openai/gpt-4o-mini")
    OPENROUTER_EMBEDDING_MODEL = os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")
    
    # Model Settings (kompatibel mit altem Code)
    @property
    def LLM_MODEL(self):
        """Returns the current LLM model (provider-dependent)"""
        if self.LLM_PROVIDER == "openrouter":
            return self.OPENROUTER_LLM_MODEL
        return self.OLLAMA_LLM_MODEL
    
    @property
    def EMBEDDING_MODEL(self):
        """Returns the current embedding model (provider-dependent)"""
        if self.LLM_PROVIDER == "openrouter":
            return self.OPENROUTER_EMBEDDING_MODEL
        return self.OLLAMA_EMBEDDING_MODEL
    
    # Retrieval Settings
    MAX_NEIGHBORS = int(os.getenv("MAX_NEIGHBORS", "5"))
    MIN_SIMILARITY_SCORE = float(os.getenv("MIN_SIMILARITY_SCORE", "0.4"))
    
    # Concurrency
    LOCK_FILE = GRAPH_DIR / "graph.lock"
    
    # TCP Server Settings (optional, für Script-Zugriff)
    TCP_SERVER_ENABLED = os.getenv("TCP_SERVER_ENABLED", "false").lower() == "true"
    TCP_SERVER_HOST = os.getenv("TCP_SERVER_HOST", "127.0.0.1")
    TCP_SERVER_PORT = int(os.getenv("TCP_SERVER_PORT", "42424"))
    
    # Researcher Agent Settings
    RESEARCHER_ENABLED = os.getenv("RESEARCHER_ENABLED", "false").lower() == "true"
    RESEARCHER_CONFIDENCE_THRESHOLD = float(os.getenv("RESEARCHER_CONFIDENCE_THRESHOLD", "0.5"))
    RESEARCHER_MAX_SOURCES = int(os.getenv("RESEARCHER_MAX_SOURCES", "5"))
    RESEARCHER_MAX_CONTENT_LENGTH = int(os.getenv("RESEARCHER_MAX_CONTENT_LENGTH", "10000"))
    
    # Jina Reader Settings (local Docker)
    JINA_READER_ENABLED = os.getenv("JINA_READER_ENABLED", "true").lower() == "true"
    JINA_READER_HOST = os.getenv("JINA_READER_HOST", "localhost")
    JINA_READER_PORT = int(os.getenv("JINA_READER_PORT", "2222"))
    JINA_READER_URL = f"http://{JINA_READER_HOST}:{JINA_READER_PORT}"
    
    # Unstructured Settings (for PDF extraction)
    UNSTRUCTURED_ENABLED = os.getenv("UNSTRUCTURED_ENABLED", "true").lower() == "true"
    UNSTRUCTURED_API_URL = os.getenv("UNSTRUCTURED_API_URL", "http://localhost:8000")
    UNSTRUCTURED_API_KEY = os.getenv("UNSTRUCTURED_API_KEY", "")
    UNSTRUCTURED_USE_LIBRARY = os.getenv("UNSTRUCTURED_USE_LIBRARY", "true").lower() == "true"  # Use library directly (default: true, falls back to API if library fails)
    
    # Google Search API Settings (for web search)
    GOOGLE_SEARCH_ENABLED = os.getenv("GOOGLE_SEARCH_ENABLED", "true").lower() == "true"
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_KEY")
    GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID", "c7bf0393f031a4691")
    
    # Cache file for amem_stats --diff
    AMEM_STATS_CACHE_FILE = DATA_DIR / "amem_stats_cache.json"
    
    # Graph Backend Selection (networkx, rustworkx, or falkordb)
    GRAPH_BACKEND = os.getenv("GRAPH_BACKEND", "networkx").lower()  # "networkx", "rustworkx", or "falkordb"

    def __init__(self):
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.GRAPH_DIR.mkdir(parents=True, exist_ok=True)

settings = Config()

