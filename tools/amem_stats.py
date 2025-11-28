#!/usr/bin/env python3
"""
A-MEM Stats CLI - Live Graph Status

Shows current memory system status, similar to `git status`.
Usage: python tools/amem_stats.py [--graph]
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from a_mem.config import settings
from a_mem.storage.engine import StorageManager
from a_mem.models.note import AtomicNote

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


def format_time_ago(timestamp_str: str) -> str:
    """Formats a timestamp as 'X min ago' or 'X hours ago'."""
    try:
        # Handle UTC timestamps (with 'Z' or without timezone info)
        if timestamp_str.endswith('Z'):
            ts = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            now = datetime.now(ts.tzinfo)  # Use UTC for comparison
        elif '+' in timestamp_str or timestamp_str.count('-') > 2:  # Has timezone
            ts = datetime.fromisoformat(timestamp_str)
            now = datetime.now(ts.tzinfo) if ts.tzinfo else datetime.utcnow()
        else:
            # No timezone info - assume UTC (for backward compatibility with old events)
            from datetime import timezone
            ts = datetime.fromisoformat(timestamp_str).replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
        
        delta = now - ts
        
        if delta.total_seconds() < 60:
            return f"{int(delta.total_seconds())}s ago"
        elif delta.total_seconds() < 3600:
            return f"{int(delta.total_seconds() / 60)}min ago"
        elif delta.total_seconds() < 86400:
            return f"{int(delta.total_seconds() / 3600)}h ago"
        else:
            return f"{delta.days}d ago"
    except Exception:
        return "unknown"


def get_last_enzyme_run() -> Optional[Dict[str, Any]]:
    """Reads the last enzyme run from events.jsonl.
    
    Reads the file backwards to find the most recent enzyme-related event.
    Looks for:
    - ENZYME_SCHEDULER_RUN (automatic scheduler runs) - preferred
    - RELATIONS_SUGGESTED (manual or automatic enzyme runs)
    - LINKS_PRUNED (manual or automatic enzyme runs)
    """
    events_file = settings.DATA_DIR / "events.jsonl"
    if not events_file.exists():
        return None
    
    # Enzyme-related events
    enzyme_events = [
        "ENZYME_SCHEDULER_RUN",
        "RELATIONS_SUGGESTED",
        "LINKS_PRUNED",
        "NODE_DIGESTED",
        "NODE_PRUNED"
    ]
    
    # Read all lines and process backwards (most recent first)
    try:
        with open(events_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        # Process backwards to find most recent event
        for line in reversed(lines):
            try:
                event = json.loads(line)
                event_type = event.get("event", "")
                
                # Prefer ENZYME_SCHEDULER_RUN or ENZYME_MANUAL_RUN (has full results)
                if event_type in ["ENZYME_SCHEDULER_RUN", "ENZYME_MANUAL_RUN"]:
                    return event
                
                # Otherwise, check for other enzyme-related events (fallback)
                if event_type in enzyme_events:
                    # For RELATIONS_SUGGESTED, this indicates a recent enzyme run
                    if event_type == "RELATIONS_SUGGESTED":
                        # Create synthetic event structure for manual runs
                        return {
                            "timestamp": event.get("timestamp", ""),
                            "event": "ENZYME_MANUAL_RUN",
                            "data": {
                                "results": {
                                    "pruned_count": 0,
                                    "zombie_nodes_removed": 0,
                                    "suggestions_count": 1,  # At least one was suggested
                                    "digested_count": 0
                                }
                            }
                        }
            except json.JSONDecodeError:
                continue
    except Exception:
        pass
    
    return None


def get_stats_from_storage() -> Dict[str, Any]:
    """Gets stats directly from StorageManager (reads from disk)."""
    manager = StorageManager()
    graph = manager.graph.graph
    
    # Count nodes by type
    type_counts = {}
    zombie_count = 0
    for node_id, attrs in graph.nodes(data=True):
        if "content" not in attrs or not str(attrs.get("content", "")).strip():
            zombie_count += 1
            continue
        node_type = attrs.get("type", "unknown")
        type_counts[node_type] = type_counts.get(node_type, 0) + 1
    
    # Count relations by type
    relation_counts = {}
    for source, target, attrs in graph.edges(data=True):
        rel_type = attrs.get("type", "relates_to")
        relation_counts[rel_type] = relation_counts.get(rel_type, 0) + 1
    
    return {
        "notes": graph.number_of_nodes() - zombie_count,
        "zombie_nodes": zombie_count,
        "relations": graph.number_of_edges(),
        "type_counts": type_counts,
        "relation_counts": relation_counts,
        "source": "disk"
    }


async def get_stats_from_http() -> Optional[Dict[str, Any]]:
    """Gets stats from running MCP server via HTTP (if available)."""
    if not HAS_AIOHTTP or not settings.TCP_SERVER_ENABLED:
        return None
    
    url = f"http://{settings.TCP_SERVER_HOST}:{settings.TCP_SERVER_PORT}/get_graph"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as response:
                if response.status == 200:
                    data = await response.json()
                    nodes = data.get("nodes", [])
                    edges = data.get("edges", [])
                    
                    # Count nodes by type
                    type_counts = {}
                    zombie_count = 0
                    for node in nodes:
                        if "content" not in node or not str(node.get("content", "")).strip():
                            zombie_count += 1
                            continue
                        node_type = node.get("type", "unknown")
                        type_counts[node_type] = type_counts.get(node_type, 0) + 1
                    
                    # Count relations by type
                    relation_counts = {}
                    for edge in edges:
                        rel_type = edge.get("relation_type", "relates_to")
                        relation_counts[rel_type] = relation_counts.get(rel_type, 0) + 1
                    
                    return {
                        "notes": len(nodes) - zombie_count,
                        "zombie_nodes": zombie_count,
                        "relations": len(edges),
                        "type_counts": type_counts,
                        "relation_counts": relation_counts,
                        "source": "http"
                    }
    except Exception:
        pass
    
    return None


def print_compact_status(stats: Dict[str, Any], last_enzyme: Optional[Dict[str, Any]] = None):
    """Prints compact one-line status."""
    notes = stats["notes"]
    relations = stats["relations"]
    zombie_nodes = stats.get("zombie_nodes", 0)
    
    time_ago = "never"
    if last_enzyme:
        timestamp = last_enzyme.get("timestamp", "")
        time_ago = format_time_ago(timestamp)
    
    zombie_str = f" | {zombie_nodes} zombies" if zombie_nodes > 0 else ""
    print(f"{notes} notes | {relations} relations | Last: {time_ago}{zombie_str}")


def print_graph_status(stats: Dict[str, Any], last_enzyme: Optional[Dict[str, Any]] = None):
    """Prints graph status in git-status style."""
    notes = stats["notes"]
    relations = stats["relations"]
    zombie_nodes = stats.get("zombie_nodes", 0)
    source = stats.get("source", "unknown")
    
    print("🧠 A-MEM Graph Status")
    print("=" * 50)
    print(f"📝 Notes:        {notes}")
    print(f"🔗 Relations:    {relations}")
    
    if zombie_nodes > 0:
        print(f"⚠️  Zombie Nodes: {zombie_nodes} (will be removed by enzymes)")
    
    # Type breakdown
    type_counts = stats.get("type_counts", {})
    if type_counts:
        print("\n📊 Notes by Type:")
        for note_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            type_emoji = {
                "rule": "🔴",
                "procedure": "🔵",
                "concept": "🟢",
                "tool": "🟠",
                "reference": "🟣",
                "integration": "🌸"
            }.get(note_type, "⚪")
            print(f"   {type_emoji} {note_type:12} {count:3}")
    
    # Relation breakdown
    relation_counts = stats.get("relation_counts", {})
    if relation_counts and len(relation_counts) <= 10:  # Only show if not too many types
        print("\n🔗 Relations by Type:")
        for rel_type, count in sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   {rel_type:20} {count:3}")
    
    # Last enzyme run
    if last_enzyme:
        timestamp = last_enzyme.get("timestamp", "")
        results = last_enzyme.get("data", {}).get("results", {})
        time_ago = format_time_ago(timestamp)
        print(f"\n⚙️  Last Enzyme Run: {time_ago}")
        if results:
            pruned = results.get("pruned_count", 0)
            zombie_removed = results.get("zombie_nodes_removed", 0)
            suggested = results.get("suggestions_count", 0)
            digested = results.get("digested_count", 0)
            if pruned > 0 or zombie_removed > 0 or suggested > 0 or digested > 0:
                print(f"   Pruned: {pruned}, Zombies removed: {zombie_removed}, "
                      f"Suggested: {suggested}, Digested: {digested}")
    else:
        print("\n⚙️  Last Enzyme Run: never")
    
    print(f"\n📡 Data Source: {source}")
    print("=" * 50)


def get_previous_stats() -> Optional[Dict[str, Any]]:
    """Loads previous stats from cache file for diff mode."""
    cache_file = settings.DATA_DIR / ".amem_stats_cache.json"
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def save_current_stats(stats: Dict[str, Any], last_enzyme: Optional[Dict[str, Any]]):
    """Saves current stats to cache file for diff mode."""
    cache_file = settings.DATA_DIR / ".amem_stats_cache.json"
    try:
        cache_data = {
            "notes": stats["notes"],
            "relations": stats["relations"],
            "zombie_nodes": stats.get("zombie_nodes", 0),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2)
    except Exception:
        pass


def print_diff_status(stats: Dict[str, Any], previous: Optional[Dict[str, Any]]):
    """Prints diff between current and previous stats."""
    if not previous:
        print("No previous stats found. This will be the baseline for future diffs.")
        return
    
    notes_diff = stats["notes"] - previous.get("notes", 0)
    relations_diff = stats["relations"] - previous.get("relations", 0)
    zombie_diff = stats.get("zombie_nodes", 0) - previous.get("zombie_nodes", 0)
    
    def format_diff(value: int) -> str:
        if value > 0:
            return f"+{value}"
        elif value < 0:
            return str(value)
        return "0"
    
    print(f"{format_diff(notes_diff)} notes | {format_diff(relations_diff)} relations | {format_diff(zombie_diff)} zombie nodes")
    
    if notes_diff == 0 and relations_diff == 0 and zombie_diff == 0:
        print("No changes since last check.")


async def main_async():
    """Async main function."""
    parser = argparse.ArgumentParser(
        description="A-MEM Graph Status - Live memory system statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/amem_stats.py              # Full status (default)
  python tools/amem_stats.py --compact    # One-line output
  python tools/amem_stats.py --json        # JSON output
  python tools/amem_stats.py --diff       # Show changes since last run
  python tools/amem_stats.py --watch      # Auto-refresh every 5 seconds
  python tools/amem_stats.py --watch 10   # Auto-refresh every 10 seconds
        """
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Compact one-line output"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        help="Show changes since last run"
    )
    parser.add_argument(
        "--watch",
        type=int,
        nargs="?",
        const=5,
        metavar="SECONDS",
        help="Watch mode: auto-refresh every N seconds (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Watch mode
    if args.watch is not None:
        import time
        import os
        import platform
        
        def clear_screen():
            if platform.system() == "Windows":
                os.system("cls")
            else:
                os.system("clear")
        
        try:
            while True:
                clear_screen()
                print(f"🧠 A-MEM Graph Status (refreshing every {args.watch}s, Ctrl+C to stop)\n")
                
                # Get stats
                stats = await get_stats_from_http()
                if stats is None:
                    stats = get_stats_from_storage()
                
                last_enzyme = get_last_enzyme_run()
                
                # Print based on mode
                if args.compact:
                    print_compact_status(stats, last_enzyme)
                elif args.json:
                    output = {
                        "notes": stats["notes"],
                        "relations": stats["relations"],
                        "zombie_nodes": stats.get("zombie_nodes", 0),
                        "type_counts": stats.get("type_counts", {}),
                        "relation_counts": stats.get("relation_counts", {}),
                        "last_enzyme_run": format_time_ago(last_enzyme.get("timestamp", "")) if last_enzyme else "never",
                        "data_source": stats.get("source", "unknown")
                    }
                    print(json.dumps(output, indent=2))
                elif args.diff:
                    previous = get_previous_stats()
                    print_diff_status(stats, previous)
                    save_current_stats(stats, last_enzyme)
                else:
                    print_graph_status(stats, last_enzyme)
                
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\n\nStopped.")
            return
    
    # Try HTTP first (live data from running server)
    stats = await get_stats_from_http()
    
    # Fallback to disk (reads from saved graph)
    if stats is None:
        stats = get_stats_from_storage()
    
    # Get last enzyme run
    last_enzyme = get_last_enzyme_run()
    
    # Print based on mode
    if args.json:
        output = {
            "notes": stats["notes"],
            "relations": stats["relations"],
            "zombie_nodes": stats.get("zombie_nodes", 0),
            "type_counts": stats.get("type_counts", {}),
            "relation_counts": stats.get("relation_counts", {}),
            "last_enzyme_run": format_time_ago(last_enzyme.get("timestamp", "")) if last_enzyme else "never",
            "data_source": stats.get("source", "unknown")
        }
        if last_enzyme and last_enzyme.get("data", {}).get("results"):
            output["last_enzyme_results"] = last_enzyme["data"]["results"]
        print(json.dumps(output, indent=2))
    elif args.compact:
        print_compact_status(stats, last_enzyme)
    elif args.diff:
        previous = get_previous_stats()
        print_diff_status(stats, previous)
        save_current_stats(stats, last_enzyme)
    else:
        # Default: full graph status
        print_graph_status(stats, last_enzyme)


def main():
    """Synchronous entry point."""
    parser = argparse.ArgumentParser(
        description="A-MEM Graph Status - Live memory system statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/amem_stats.py              # Full status (default)
  python tools/amem_stats.py --compact    # One-line output
  python tools/amem_stats.py --json       # JSON output
  python tools/amem_stats.py --diff       # Show changes since last run
  python tools/amem_stats.py --watch      # Auto-refresh every 5 seconds
        """
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Compact one-line output"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        help="Show changes since last run"
    )
    parser.add_argument(
        "--watch",
        type=int,
        nargs="?",
        const=5,
        metavar="SECONDS",
        help="Watch mode: auto-refresh every N seconds (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Watch mode (sync version)
    if args.watch is not None:
        import time
        import os
        import platform
        
        def clear_screen():
            if platform.system() == "Windows":
                os.system("cls")
            else:
                os.system("clear")
        
        try:
            while True:
                clear_screen()
                print(f"🧠 A-MEM Graph Status (refreshing every {args.watch}s, Ctrl+C to stop)\n")
                
                stats = get_stats_from_storage()
                last_enzyme = get_last_enzyme_run()
                
                if args.compact:
                    print_compact_status(stats, last_enzyme)
                elif args.json:
                    output = {
                        "notes": stats["notes"],
                        "relations": stats["relations"],
                        "zombie_nodes": stats.get("zombie_nodes", 0),
                        "type_counts": stats.get("type_counts", {}),
                        "relation_counts": stats.get("relation_counts", {}),
                        "last_enzyme_run": format_time_ago(last_enzyme.get("timestamp", "")) if last_enzyme else "never",
                        "data_source": stats.get("source", "unknown")
                    }
                    print(json.dumps(output, indent=2))
                elif args.diff:
                    previous = get_previous_stats()
                    print_diff_status(stats, previous)
                    save_current_stats(stats, last_enzyme)
                else:
                    print_graph_status(stats, last_enzyme)
                
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\n\nStopped.")
            return
    
    if HAS_AIOHTTP:
        import asyncio
        asyncio.run(main_async())
    else:
        # Fallback: only use disk
        stats = get_stats_from_storage()
        last_enzyme = get_last_enzyme_run()
        
        if args.json:
            output = {
                "notes": stats["notes"],
                "relations": stats["relations"],
                "zombie_nodes": stats.get("zombie_nodes", 0),
                "type_counts": stats.get("type_counts", {}),
                "relation_counts": stats.get("relation_counts", {}),
                "last_enzyme_run": format_time_ago(last_enzyme.get("timestamp", "")) if last_enzyme else "never",
                "data_source": stats.get("source", "unknown")
            }
            print(json.dumps(output, indent=2))
        elif args.compact:
            print_compact_status(stats, last_enzyme)
        elif args.diff:
            previous = get_previous_stats()
            print_diff_status(stats, previous)
            save_current_stats(stats, last_enzyme)
        else:
            print_graph_status(stats, last_enzyme)


if __name__ == "__main__":
    main()

