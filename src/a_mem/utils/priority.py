"""
Priority Scoring and Event Logging Utilities

KISS-Approach: Priority is computed on-the-fly, not persisted.
Event logging uses append-only JSONL format.
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
from ..models.note import AtomicNote
from ..config import settings

# Event Log Path
EVENT_LOG_PATH = settings.DATA_DIR / "events.jsonl"


def compute_priority(
    note: AtomicNote,
    usage_count: int = 0,
    edge_count: int = 0
) -> float:
    """
    Computes priority score for a note (on-the-fly, not persistent).
    
    Formula: priority = type_weight * (1 + usage_factor) * (1 + edge_factor) / age_factor
    
    Args:
        note: The note to score
        usage_count: How often this note was retrieved
        edge_count: Number of graph connections (degree)
    
    Returns:
        Priority score (higher = more important)
    """
    # Type weights (rules and procedures are more important)
    type_weights = {
        "rule": 1.4,
        "procedure": 1.3,
        "concept": 1.1,
        "tool": 1.0,
        "reference": 0.9,
        "integration": 1.0,
        None: 1.0
    }
    
    type_weight = type_weights.get(note.type, 1.0)
    
    # Age factor (older = less weight, but never below 0.5)
    age_days = (datetime.now() - note.created_at).days + 1
    age_factor = max(0.5, 1.0 - age_days * 0.005)  # Slower decay
    
    # Usage factor (more usage = higher priority)
    usage_factor = 1 + (usage_count * 0.2)
    
    # Edge factor (more connections = higher priority)
    edge_factor = 1 + (edge_count * 0.1)  # Increased from 0.05 to 0.1
    
    # Final calculation: Multiply age_factor (not divide) so older = lower priority
    priority = type_weight * usage_factor * edge_factor * age_factor
    
    return round(priority, 3)


def log_event(event_type: str, data: Dict[str, Any]) -> None:
    """
    Logs an event to append-only JSONL file.
    
    Args:
        event_type: Type of event (e.g., "NOTE_CREATED", "RELATION_CREATED")
        data: Event data dictionary
    """
    # Ensure log directory exists
    EVENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",  # Add 'Z' suffix for UTC
        "event": event_type,
        "data": data
    }
    
    try:
        with open(EVENT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        import sys
        print(f"Event logging error: {e}", file=sys.stderr)


def keyword_prefilter(query: str, notes: list[AtomicNote]) -> list[AtomicNote]:
    """
    Pre-filters notes by keyword matching (cheap operation before LLM).
    
    Args:
        query: Search query string
        notes: List of candidate notes
    
    Returns:
        Filtered list of notes containing query keywords
    """
    if not query or not notes:
        return notes
    
    query_tokens = set(query.lower().split())
    
    filtered = []
    for note in notes:
        # Check content, summary, keywords, and tags
        searchable_text = (
            f"{note.content} {note.contextual_summary} "
            f"{' '.join(note.keywords)} {' '.join(note.tags)}"
        ).lower()
        
        # Match if any query token appears in searchable text
        if any(token in searchable_text for token in query_tokens):
            filtered.append(note)
    
    return filtered

