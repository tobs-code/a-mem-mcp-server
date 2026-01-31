"""
Test script for FalkorDBLite GraphStore Adapter (Proof of Concept)

Run with: python -m pytest tests/test_falkordb_store.py -v
Or: python tests/test_falkordb_store.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from datetime import datetime
from a_mem.storage.falkordb_store import FalkorDBGraphStore, FALKORDB_AVAILABLE
from a_mem.models.note import AtomicNote, NoteRelation


def test_falkordb_basic_operations():
    """Test basic CRUD operations with FalkorDBLite."""
    
    if not FALKORDB_AVAILABLE:
        print("SKIP: FalkorDBLite not installed. Install with: pip install falkordblite")
        return
    
    # Initialize store
    store = FalkorDBGraphStore()
    
    # Reset for clean test
    store.reset()
    
    # Test 1: Add node
    note1 = AtomicNote(
        id="test-node-1",
        content="Test content 1",
        contextual_summary="Test summary 1",
        keywords=["test", "keyword"],
        tags=["tag1"],
        type="concept"
    )
    store.add_node(note1)
    print("✅ Test 1: Added node")
    
    # Test 2: Get node
    retrieved = store.get_node("test-node-1")
    assert retrieved is not None, "Node should exist"
    assert retrieved["id"] == "test-node-1", "Node ID should match"
    assert retrieved["content"] == "Test content 1", "Content should match"
    print("✅ Test 2: Retrieved node")
    
    # Test 3: Has node
    assert store.has_node("test-node-1"), "Node should exist"
    assert not store.has_node("non-existent"), "Non-existent node should not exist"
    print("✅ Test 3: Has node check")
    
    # Test 4: Add second node
    note2 = AtomicNote(
        id="test-node-2",
        content="Test content 2",
        contextual_summary="Test summary 2",
        keywords=["test", "keyword2"],
        tags=["tag2"],
        type="rule"
    )
    store.add_node(note2)
    print("✅ Test 4: Added second node")
    
    # Test 5: Add edge
    relation = NoteRelation(
        source_id="test-node-1",
        target_id="test-node-2",
        relation_type="relates_to",
        reasoning="Test relation",
        weight=0.8
    )
    store.add_edge(relation)
    print("✅ Test 5: Added edge")
    
    # Test 6: Get neighbors
    neighbors = store.get_neighbors("test-node-1")
    assert len(neighbors) > 0, "Should have neighbors"
    assert any(n.get("id") == "test-node-2" for n in neighbors), "Should find connected node"
    print("✅ Test 6: Retrieved neighbors")
    
    # Test 7: Update node
    note1_updated = AtomicNote(
        id="test-node-1",
        content="Updated content 1",
        contextual_summary="Updated summary 1",
        keywords=["test", "updated"],
        tags=["tag1", "updated"],
        type="concept"
    )
    store.update_node(note1_updated)
    retrieved_updated = store.get_node("test-node-1")
    assert retrieved_updated["content"] == "Updated content 1", "Content should be updated"
    print("✅ Test 7: Updated node")
    
    # Test 8: Get all nodes
    all_nodes = store.get_all_nodes()
    assert len(all_nodes) >= 2, "Should have at least 2 nodes"
    print(f"✅ Test 8: Retrieved all nodes ({len(all_nodes)} nodes)")
    
    # Test 9: Get all edges
    all_edges = store.get_all_edges()
    assert len(all_edges) >= 1, "Should have at least 1 edge"
    print(f"✅ Test 9: Retrieved all edges ({len(all_edges)} edges)")
    
    # Test 10: Remove node (should also remove edges)
    store.remove_node("test-node-1")
    assert not store.has_node("test-node-1"), "Node should be removed"
    print("✅ Test 10: Removed node")
    
    # Test 11: Reset
    store.reset()
    all_nodes_after_reset = store.get_all_nodes()
    assert len(all_nodes_after_reset) == 0, "Graph should be empty after reset"
    print("✅ Test 11: Reset graph")
    
    print("\n🎉 All tests passed!")


if __name__ == "__main__":
    test_falkordb_basic_operations()










