"""
Memory Enzymes: Autonome Hintergrund-Prozesse für Graph-Pflege

KISS-Approach: Kleine, unabhängige Module die den Graph automatisch optimieren.
- prune_links: Entfernt alte/schwache Links
- suggest_relations: Schlägt neue Verbindungen vor
- digest_node: Komprimiert überfüllte Nodes
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import sys
from ..models.note import AtomicNote, NoteRelation
from ..storage.engine import GraphStore
from ..utils.llm import LLMService
from ..utils.priority import log_event

# Helper function to print to stderr (MCP uses stdout for JSON-RPC)
def log_debug(message: str):
    """Logs debug messages to stderr to avoid breaking MCP JSON-RPC on stdout."""
    print(message, file=sys.stderr)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Berechnet Cosine Similarity zwischen zwei Embeddings."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    dot_product = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


def prune_links(
    graph: GraphStore,
    max_age_days: int = 90,
    min_weight: float = 0.3,
    min_usage: int = 0
) -> int:
    """
    Entfernt schwache oder alte Kanten aus dem Graph.
    
    Args:
        graph: GraphStore Instanz
        max_age_days: Maximale Alter in Tagen (default: 90)
        min_weight: Minimale Edge-Weight (default: 0.3)
        min_usage: Minimale Usage-Count (default: 0, da wir das noch nicht tracken)
    
    Returns:
        Anzahl entfernte Edges
    """
    now = datetime.utcnow()
    to_remove = []
    
    for source, target, data in graph.graph.edges(data=True):
        should_remove = False
        
        # CRITICAL: Orphaned Edge Check - Entferne Edges zu nicht existierenden Nodes
        if source not in graph.graph.nodes or target not in graph.graph.nodes:
            to_remove.append((source, target))
            log_event("RELATION_PRUNED", {
                "source": source,
                "target": target,
                "reason": "orphaned_edge_missing_node"
            })
            continue  # Skip weitere Checks, Edge wird entfernt
        
        # CRITICAL: Zombie Node Check - Entferne Edges zu Nodes ohne Content (gelöschte Nodes)
        # Handle None data gracefully
        if data is None:
            data = {}
        source_node = graph.graph.nodes[source]
        target_node = graph.graph.nodes[target]
        
        # Prüfe ob Nodes leer sind (keine Keys) oder kein Content haben
        source_is_empty = len(source_node) == 0 or "content" not in source_node
        target_is_empty = len(target_node) == 0 or "content" not in target_node
        
        if not source_is_empty:
            source_content = source_node.get("content", "")
            source_has_content = source_content and len(str(source_content).strip()) > 0
        else:
            source_has_content = False
        
        if not target_is_empty:
            target_content = target_node.get("content", "")
            target_has_content = target_content and len(str(target_content).strip()) > 0
        else:
            target_has_content = False
        
        if not source_has_content or not target_has_content:
            to_remove.append((source, target))
            log_event("RELATION_PRUNED", {
                "source": source,
                "target": target,
                "reason": "orphaned_edge_zombie_node",
                "source_is_empty": source_is_empty,
                "target_is_empty": target_is_empty,
                "source_has_content": source_has_content,
                "target_has_content": target_has_content
            })
            continue  # Skip weitere Checks, Edge wird entfernt
        
        # Weight-Check: Schwache Verbindungen entfernen
        weight = data.get("weight", 1.0)
        if weight < min_weight:
            should_remove = True
        
        # Age-Check: Alte Verbindungen entfernen
        if "created_at" in data:
            try:
                # created_at ist als ISO string gespeichert
                edge_time = datetime.fromisoformat(data["created_at"])
                age_days = (now - edge_time).days
                if age_days > max_age_days:
                    should_remove = True
            except (ValueError, TypeError):
                # Fallback: Prüfe Node-Alter wenn Edge kein created_at hat
                if source in graph.graph.nodes and target in graph.graph.nodes:
                    source_node = graph.graph.nodes[source]
                    target_node = graph.graph.nodes[target]
                    
                    # Wenn beide Nodes alt sind und Edge schwach ist → entfernen
                    source_created = source_node.get("created_at")
                    target_created = target_node.get("created_at")
                    
                    if isinstance(source_created, str):
                        try:
                            source_created = datetime.fromisoformat(source_created)
                        except ValueError:
                            source_created = None
                    
                    if isinstance(target_created, str):
                        try:
                            target_created = datetime.fromisoformat(target_created)
                        except ValueError:
                            target_created = None
                    
                    if source_created and target_created:
                        source_age = (now - source_created).days
                        target_age = (now - target_created).days
                        if source_age > max_age_days and target_age > max_age_days and weight < 0.5:
                            should_remove = True
        
        if should_remove:
            to_remove.append((source, target))
    
    # Entferne Edges
    for source, target in to_remove:
        graph.graph.remove_edge(source, target)
    
    if to_remove:
        log_event("LINKS_PRUNED", {
            "count": len(to_remove),
            "max_age_days": max_age_days,
            "min_weight": min_weight
        })
    
    return len(to_remove)


def prune_zombie_nodes(graph: GraphStore) -> int:
    """
    Entfernt Zombie-Nodes (Nodes ohne Content) aus dem Graph.
    
    Args:
        graph: GraphStore Instanz
    
    Returns:
        Anzahl entfernte Nodes
    """
    to_remove = []
    
    for node_id in list(graph.graph.nodes()):
        node_data = graph.graph.nodes[node_id]
        
        # Prüfe ob Node leer ist (keine Keys) oder kein Content hat
        is_empty = len(node_data) == 0 or "content" not in node_data
        
        if not is_empty:
            content = node_data.get("content", "")
            has_content = content and len(str(content).strip()) > 0
        else:
            has_content = False
        
        if not has_content:
            to_remove.append(node_id)
            log_event("NODE_PRUNED", {
                "node_id": node_id,
                "reason": "zombie_node_no_content",
                "is_empty": is_empty
            })
    
    # Entferne Nodes
    for node_id in to_remove:
        graph.graph.remove_node(node_id)
    
    if to_remove:
        log_event("ZOMBIE_NODES_PRUNED", {
            "count": len(to_remove),
            "node_ids": to_remove[:10]  # Nur erste 10 loggen
        })
    
    return len(to_remove)


def merge_duplicates(
    notes: Dict[str, AtomicNote],
    graph: GraphStore,
    llm_service: Optional[LLMService] = None,
    content_similarity_threshold: float = 0.98
) -> int:
    """
    Findet und merged Duplikate (Notes mit identischem oder sehr ähnlichem Content).
    
    Strategie:
    1. Finde Notes mit identischem Content (exakt)
    2. Finde Notes mit semantisch ähnlichem Content (via Embeddings, wenn llm_service verfügbar)
    3. Bei exakten Duplikaten: Behalte beste Note, lösche andere
    4. Bei semantischen Duplikaten: Intelligentes Merging (Content zusammenführen)
    5. Leite alle Edges von Duplikaten auf die behaltene Note um
    
    Args:
        notes: Dict von note_id -> AtomicNote
        graph: GraphStore Instanz
        llm_service: Optional LLMService für Embedding-Berechnung (für semantische Duplikate)
        content_similarity_threshold: Minimale Content-Similarity für semantische Duplikat-Erkennung (default: 0.98)
    
    Returns:
        Anzahl gemergte Duplikate
    """
    if len(notes) < 2:
        return 0
    
    merged_count = 0
    note_ids = list(notes.keys())
    to_remove = set()  # IDs von Notes die gelöscht werden sollen
    
    # Phase 1: Exakte Duplikate (identischer Content)
    for i in range(len(note_ids)):
        if note_ids[i] in to_remove:
            continue
            
        for j in range(i + 1, len(note_ids)):
            if note_ids[j] in to_remove:
                continue
            
            note_a = notes[note_ids[i]]
            note_b = notes[note_ids[j]]
            
            # Exakte Content-Übereinstimmung
            if note_a.content.strip() == note_b.content.strip() and note_a.content.strip():
                # Entscheide welche Note behalten wird
                keep_id, remove_id = _choose_best_note(
                    note_ids[i], note_ids[j], note_a, note_b, graph
                )
                
                if keep_id and remove_id:
                    # Leite Edges um
                    _redirect_edges(graph, remove_id, keep_id)
                    
                    # Markiere zum Löschen
                    to_remove.add(remove_id)
                    merged_count += 1
                    
                    log_event("DUPLICATE_MERGED", {
                        "kept": keep_id,
                        "removed": remove_id,
                        "reason": "identical_content"
                    })
    
    # Phase 1.5: Identische Summaries (schneller Check, kein Embedding nötig)
    for i in range(len(note_ids)):
        if note_ids[i] in to_remove:
            continue
            
        for j in range(i + 1, len(note_ids)):
            if note_ids[j] in to_remove:
                continue
            
            note_a = notes[note_ids[i]]
            note_b = notes[note_ids[j]]
            
            # Identische Summary-Übereinstimmung (und gleiche Quelle wenn vorhanden)
            summary_a = (note_a.contextual_summary or "").strip()
            summary_b = (note_b.contextual_summary or "").strip()
            
            if summary_a and summary_a == summary_b:
                # Zusätzliche Prüfung: Gleiche Quelle oder sehr ähnlicher Content
                source_a = note_a.metadata.get("source_url", "") if note_a.metadata else ""
                source_b = note_b.metadata.get("source_url", "") if note_b.metadata else ""
                
                # Wenn gleiche Quelle ODER sehr ähnlicher Content (erste 100 Zeichen)
                if source_a == source_b or (len(note_a.content) > 50 and len(note_b.content) > 50 and 
                                            note_a.content[:100] == note_b.content[:100]):
                    # Intelligentes Merging
                    keep_id, remove_id = _choose_best_note(
                        note_ids[i], note_ids[j], note_a, note_b, graph
                    )
                    
                    if keep_id and remove_id:
                        # Merge Content intelligent
                        merged_note = _merge_note_content(
                            notes[keep_id], notes[remove_id]
                        )
                        
                        # Aktualisiere Note im Graph
                        if keep_id in graph.graph.nodes:
                            node_data = graph.graph.nodes[keep_id]
                            node_data.update({
                                "content": merged_note.content,
                                "contextual_summary": merged_note.contextual_summary,
                                "keywords": list(set(merged_note.keywords)),
                                "tags": list(set(merged_note.tags))
                            })
                        
                        # Leite Edges um
                        _redirect_edges(graph, remove_id, keep_id)
                        
                        # Markiere zum Löschen
                        to_remove.add(remove_id)
                        merged_count += 1
                        
                        log_event("DUPLICATE_MERGED", {
                            "kept": keep_id,
                            "removed": remove_id,
                            "reason": "identical_summary",
                            "summary": summary_a[:50] + "..." if len(summary_a) > 50 else summary_a
                        })
    
    # Phase 2: Semantische Duplikate (via Embeddings, wenn llm_service verfügbar)
    if llm_service and len(notes) >= 2:
        # Berechne Embeddings für alle Notes (einmalig)
        content_embeddings = {}
        for note_id in note_ids:
            if note_id in to_remove:
                continue
            note = notes[note_id]
            try:
                # Verwende Content + Summary für Embedding
                text_for_embedding = f"{note.content} {note.contextual_summary or ''}"
                embedding = llm_service.get_embedding(text_for_embedding)
                content_embeddings[note_id] = embedding
            except Exception as e:
                log_debug(f"Error computing embedding for {note_id} in merge_duplicates: {e}")
                continue
        
        # Finde semantische Duplikate
        for i in range(len(note_ids)):
            if note_ids[i] in to_remove or note_ids[i] not in content_embeddings:
                continue
                
            for j in range(i + 1, len(note_ids)):
                if note_ids[j] in to_remove or note_ids[j] not in content_embeddings:
                    continue
                
                note_a = notes[note_ids[i]]
                note_b = notes[note_ids[j]]
                
                # Berechne Similarity
                similarity = cosine_similarity(
                    content_embeddings[note_ids[i]],
                    content_embeddings[note_ids[j]]
                )
                
                # Prüfe ob semantisches Duplikat
                if similarity >= content_similarity_threshold:
                    # Intelligentes Merging statt einfachem Löschen
                    keep_id, remove_id = _choose_best_note(
                        note_ids[i], note_ids[j], note_a, note_b, graph
                    )
                    
                    if keep_id and remove_id:
                        # Merge Content intelligent
                        merged_note = _merge_note_content(
                            notes[keep_id], notes[remove_id]
                        )
                        
                        # Aktualisiere Note im Graph
                        if keep_id in graph.graph.nodes:
                            node_data = graph.graph.nodes[keep_id]
                            node_data.update({
                                "content": merged_note.content,
                                "contextual_summary": merged_note.contextual_summary,
                                "keywords": list(set(merged_note.keywords)),
                                "tags": list(set(merged_note.tags))
                            })
                        
                        # Leite Edges um
                        _redirect_edges(graph, remove_id, keep_id)
                        
                        # Markiere zum Löschen
                        to_remove.add(remove_id)
                        merged_count += 1
                        
                        log_event("DUPLICATE_MERGED", {
                            "kept": keep_id,
                            "removed": remove_id,
                            "reason": "semantic_similarity",
                            "similarity": similarity
                        })
    
    # Entferne Duplikate
    for node_id in to_remove:
        if node_id in graph.graph.nodes:
            graph.graph.remove_node(node_id)
    
    if merged_count > 0:
        log_event("DUPLICATES_MERGED", {
            "count": merged_count,
            "removed_ids": list(to_remove)[:10]  # Nur erste 10 loggen
        })
    
    return merged_count


def _merge_note_content(note_a: AtomicNote, note_b: AtomicNote) -> AtomicNote:
    """
    Merged zwei Notes intelligent: Kombiniert Content, Summary, Keywords und Tags.
    
    Strategie:
    - Content: Kombiniere beide, entferne Duplikate
    - Summary: Nimm die längere/detailliertere
    - Keywords: Vereinige beide Sets
    - Tags: Vereinige beide Sets
    - Metadata: Kombiniere beide (keine Duplikate)
    
    Args:
        note_a: Erste Note (wird als Basis verwendet)
        note_b: Zweite Note (wird in note_a integriert)
    
    Returns:
        Gemergte Note (basierend auf note_a)
    """
    # Content: Kombiniere beide, entferne identische Absätze
    content_a = note_a.content.strip()
    content_b = note_b.content.strip()
    
    # Teile Content in Absätze
    paragraphs_a = [p.strip() for p in content_a.split('\n\n') if p.strip()]
    paragraphs_b = [p.strip() for p in content_b.split('\n\n') if p.strip()]
    
    # Kombiniere, entferne Duplikate
    merged_paragraphs = paragraphs_a.copy()
    for para_b in paragraphs_b:
        # Prüfe ob Absatz bereits vorhanden (exakt oder sehr ähnlich)
        is_duplicate = False
        for para_a in paragraphs_a:
            # Exakte Übereinstimmung oder sehr ähnlich (gleiche Länge, ähnlicher Anfang)
            if para_b == para_a or (len(para_b) > 50 and para_b[:50] == para_a[:50]):
                is_duplicate = True
                break
        if not is_duplicate:
            merged_paragraphs.append(para_b)
    
    merged_content = '\n\n'.join(merged_paragraphs)
    
    # Summary: Nimm die längere/detailliertere
    summary_a = note_a.contextual_summary or ""
    summary_b = note_b.contextual_summary or ""
    merged_summary = summary_a if len(summary_a) >= len(summary_b) else summary_b
    
    # Keywords: Vereinige beide Sets
    merged_keywords = list(set(note_a.keywords + note_b.keywords))
    
    # Tags: Vereinige beide Sets
    merged_tags = list(set(note_a.tags + note_b.tags))
    
    # Metadata: Kombiniere beide (note_a als Basis)
    merged_metadata = note_a.metadata.copy() if note_a.metadata else {}
    if note_b.metadata:
        for key, value in note_b.metadata.items():
            # Überschreibe nur wenn key nicht existiert oder value informativer ist
            if key not in merged_metadata or (isinstance(value, str) and len(str(value)) > len(str(merged_metadata.get(key, "")))):
                merged_metadata[key] = value
    
    # Erstelle gemergte Note (basierend auf note_a)
    merged_note = AtomicNote(
        id=note_a.id,
        content=merged_content,
        contextual_summary=merged_summary,
        keywords=merged_keywords,
        tags=merged_tags,
        created_at=note_a.created_at,  # Behalte ursprüngliches created_at
        type=note_a.type,
        metadata=merged_metadata
    )
    
    return merged_note


def _choose_best_note(
    id_a: str, id_b: str, note_a: AtomicNote, note_b: AtomicNote, graph: GraphStore
) -> Tuple[Optional[str], Optional[str]]:
    """
    Entscheidet welche Note behalten wird basierend auf Qualitätskriterien.
    
    Returns:
        (keep_id, remove_id) oder (None, None) wenn unentscheidbar
    """
    score_a = 0
    score_b = 0
    
    # Kriterium 1: Metadaten (mehr = besser)
    if note_a.metadata:
        score_a += len(note_a.metadata)
    if note_b.metadata:
        score_b += len(note_b.metadata)
    
    # Kriterium 2: Summary-Qualität (länger/spezifischer = besser)
    if note_a.contextual_summary:
        score_a += len(note_a.contextual_summary)
    if note_b.contextual_summary:
        score_b += len(note_b.contextual_summary)
    
    # Kriterium 3: Anzahl Verbindungen (mehr = besser)
    edges_a = graph.graph.degree(id_a) if id_a in graph.graph.nodes else 0
    edges_b = graph.graph.degree(id_b) if id_b in graph.graph.nodes else 0
    score_a += edges_a * 2  # Gewichtung: Verbindungen sind wichtig
    score_b += edges_b * 2
    
    # Kriterium 4: Keywords/Tags (mehr = besser)
    score_a += len(note_a.keywords) + len(note_a.tags)
    score_b += len(note_b.keywords) + len(note_b.tags)
    
    # Entscheidung
    if score_a > score_b:
        return id_a, id_b
    elif score_b > score_a:
        return id_b, id_a
    else:
        # Bei Gleichstand: Ältere Note behalten (mehr Zeit für Evolution)
        if note_a.created_at < note_b.created_at:
            return id_a, id_b
        else:
            return id_b, id_a


def _redirect_edges(graph: GraphStore, from_id: str, to_id: str):
    """
    Leitet alle Edges von from_id auf to_id um.
    """
    if from_id not in graph.graph.nodes:
        return
    
    # Sammle alle Edges die umgeleitet werden müssen
    edges_to_redirect = []
    
    # Outgoing edges (from_id -> target)
    for target in list(graph.graph.successors(from_id)):
        edge_data = graph.graph.get_edge_data(from_id, target)
        if edge_data:
            # Kopiere Edge-Daten (get_edge_data gibt Dict zurück)
            edge_copy = dict(edge_data) if isinstance(edge_data, dict) else edge_data
            edges_to_redirect.append(("out", target, edge_copy))
    
    # Incoming edges (source -> from_id)
    for source in list(graph.graph.predecessors(from_id)):
        edge_data = graph.graph.get_edge_data(source, from_id)
        if edge_data:
            # Kopiere Edge-Daten
            edge_copy = dict(edge_data) if isinstance(edge_data, dict) else edge_data
            edges_to_redirect.append(("in", source, edge_copy))
    
    # Erstelle neue Edges und entferne alte
    for direction, other_id, edge_data in edges_to_redirect:
        try:
            if direction == "out":
                # from_id -> target wird zu to_id -> target
                if not graph.graph.has_edge(to_id, other_id):
                    graph.graph.add_edge(to_id, other_id, **edge_data)
                graph.graph.remove_edge(from_id, other_id)
            else:
                # source -> from_id wird zu source -> to_id
                if not graph.graph.has_edge(other_id, to_id):
                    graph.graph.add_edge(other_id, to_id, **edge_data)
                graph.graph.remove_edge(other_id, from_id)
        except Exception as e:
            log_debug(f"Error redirecting edge {from_id} -> {to_id}: {e}")
            continue
    
    log_event("EDGES_REDIRECTED", {
        "from": from_id,
        "to": to_id,
        "edges_count": len(edges_to_redirect)
    })


def validate_and_fix_edges(
    graph: GraphStore,
    notes: Dict[str, AtomicNote],
    llm_service: Optional[LLMService] = None,
    min_weight_for_reasoning: float = 0.65,
    max_flag_age_days: int = 30,
    ignore_flags: bool = False
) -> Dict[str, Any]:
    """
    Validiert und korrigiert Edges:
    - Entfernt Edges mit leeren Reasoning-Feldern (wenn weight < threshold)
    - Ergänzt fehlende Reasoning-Felder via LLM
    - Standardisiert Relation Types (similar_to → relates_to)
    - Entfernt schwache/unsinnige Edges
    
    Args:
        graph: GraphStore Instanz
        notes: Dict von note_id -> AtomicNote
        llm_service: Optional LLMService für Reasoning-Generierung
        min_weight_for_reasoning: Minimale Weight für Edges die Reasoning benötigen (default: 0.65)
    
    Returns:
        Dict mit Ergebnissen:
        - "edges_removed": Anzahl entfernte Edges
        - "reasonings_added": Anzahl ergänzte Reasoning-Felder
        - "types_standardized": Anzahl standardisierte Relation Types
    """
    edges_removed = 0
    reasonings_added = 0
    types_standardized = 0
    to_remove = []
    to_update = []
    
    for source, target, data in graph.graph.edges(data=True):
        # Handle None data gracefully
        if data is None:
            data = {}
        if source not in graph.graph.nodes or target not in graph.graph.nodes:
            continue
        
        # Prüfe ob Nodes existieren
        if source not in notes or target not in notes:
            continue
        
        source_note = notes[source]
        target_note = notes[target]
        
        # 1. Standardisiere Relation Types (similar_to → relates_to)
        relation_type = data.get("type") or data.get("relation_type", "relates_to")
        if relation_type == "similar_to":
            data["type"] = "relates_to"
            if "relation_type" in data:
                del data["relation_type"]
            types_standardized += 1
            to_update.append((source, target, data))
        
        # 2. Standardisiere Edge-Felder (type vs relation_type)
        if "relation_type" in data and "type" not in data:
            data["type"] = data["relation_type"]
            del data["relation_type"]
            types_standardized += 1
            to_update.append((source, target, data))
        
        # 3. Prüfe Reasoning-Feld
        reasoning = data.get("reasoning", "")
        weight = data.get("weight", 1.0)
        
        # Prüfe auf widersprüchliche Reasoning (z.B. "Keine Beziehung" aber weight hoch)
        if reasoning and len(reasoning.strip()) >= 10:
            reasoning_lower = reasoning.lower()
            negative_phrases = ["keine beziehung", "no relationship", "nicht verwandt", "not related", "unrelated", "keine direkte beziehung", "no direct relationship"]
            if any(phrase in reasoning_lower for phrase in negative_phrases):
                if weight > 0.7:
                    # Widerspruch: Reasoning sagt "keine Beziehung" aber weight ist hoch → entfernen
                    to_remove.append((source, target))
                    edges_removed += 1
                    log_event("EDGE_REMOVED_CONTRADICTORY", {
                        "source": source,
                        "target": target,
                        "weight": weight,
                        "reasoning": reasoning[:50],
                        "reason": "contradictory_reasoning_high_weight"
                    })
                    continue
                elif weight < 0.7:
                    # Schwache Edge mit negativem Reasoning → entfernen
                    to_remove.append((source, target))
                    edges_removed += 1
                    log_event("EDGE_REMOVED_WEAK_NEGATIVE", {
                        "source": source,
                        "target": target,
                        "weight": weight,
                        "reasoning": reasoning[:50],
                        "reason": "weak_edge_negative_reasoning"
                    })
                    continue
        
        # Wenn Reasoning leer und Weight niedrig → entfernen oder Reasoning ergänzen
        if not reasoning or len(reasoning.strip()) < 10:
            if weight < min_weight_for_reasoning:
                # Edge ist schwach und hat kein Reasoning → entfernen
                to_remove.append((source, target))
                edges_removed += 1
                log_event("EDGE_REMOVED_NO_REASONING", {
                    "source": source,
                    "target": target,
                    "weight": weight,
                    "reason": "low_weight_no_reasoning"
                })
                continue
            elif llm_service:
                # Edge hat Weight aber kein Reasoning → generiere Reasoning
                try:
                    prompt = f"""Erkläre die Beziehung zwischen diesen beiden Notes:

Note A (Source):
Summary: {source_note.contextual_summary}
Keywords: {', '.join(source_note.keywords[:5])}

Note B (Target):
Summary: {target_note.contextual_summary}
Keywords: {', '.join(target_note.keywords[:5])}

Erstelle eine kurze, prägnante Erklärung (max 150 Zeichen) warum diese Notes verbunden sind.
Antworte nur mit der Erklärung, keine zusätzlichen Worte."""
                    
                    new_reasoning = llm_service._call_llm(prompt).strip()
                    if new_reasoning and len(new_reasoning) > 10:
                        data["reasoning"] = new_reasoning[:150]
                        reasonings_added += 1
                        to_update.append((source, target, data))
                        log_event("EDGE_REASONING_ADDED", {
                            "source": source,
                            "target": target,
                            "reasoning": new_reasoning[:50]
                        })
                except Exception as e:
                    log_debug(f"Error generating reasoning for {source} → {target}: {e}")
        
        # 4. Prüfe ob Edge sinnvoll ist (basierend auf Content-Ähnlichkeit)
        # Wenn Weight sehr niedrig (< 0.5) und keine gemeinsamen Keywords/Tags → entfernen
        if weight < 0.5:
            common_keywords = set(source_note.keywords) & set(target_note.keywords)
            common_tags = set(source_note.tags) & set(target_note.tags)
            
            if not common_keywords and not common_tags:
                # Keine Gemeinsamkeiten und niedrige Weight → entfernen
                to_remove.append((source, target))
                edges_removed += 1
                log_event("EDGE_REMOVED_LOW_SIMILARITY", {
                    "source": source,
                    "target": target,
                    "weight": weight,
                    "reason": "low_weight_no_common_keywords_tags"
                })
    
    # Entferne Edges
    for source, target in to_remove:
        if graph.graph.has_edge(source, target):
            graph.graph.remove_edge(source, target)
    
    # Update Edges mit neuen Daten
    for source, target, data in to_update:
        if graph.graph.has_edge(source, target):
            # Update Edge-Daten
            for key, value in data.items():
                graph.graph[source][target][key] = value
    
    if edges_removed > 0 or reasonings_added > 0 or types_standardized > 0:
        log_event("EDGES_VALIDATED", {
            "edges_removed": edges_removed,
            "reasonings_added": reasonings_added,
            "types_standardized": types_standardized
        })
    
    return {
        "edges_removed": edges_removed,
        "reasonings_added": reasonings_added,
        "types_standardized": types_standardized
    }


def remove_self_loops(graph: GraphStore) -> int:
    """
    Entfernt Self-Loops (Edges von einem Node zu sich selbst).
    
    Args:
        graph: GraphStore Instanz
    
    Returns:
        Anzahl entfernte Self-Loops
    """
    self_loops = []
    
    for node_id in graph.graph.nodes():
        if graph.graph.has_edge(node_id, node_id):
            self_loops.append(node_id)
            graph.graph.remove_edge(node_id, node_id)
    
    if self_loops:
        log_event("SELF_LOOPS_REMOVED", {
            "count": len(self_loops),
            "node_ids": self_loops[:10]  # Nur erste 10 loggen
        })
    
    return len(self_loops)


def find_isolated_nodes(
    notes: Dict[str, AtomicNote],
    graph: GraphStore
) -> List[str]:
    """
    Findet isolierte Nodes (keine Verbindungen).
    
    Args:
        notes: Dict von note_id -> AtomicNote
        graph: GraphStore Instanz
    
    Returns:
        Liste von isolierten Node-IDs
    """
    isolated = []
    
    for node_id in notes.keys():
        if node_id not in graph.graph.nodes:
            continue
        
        # Prüfe ob Node Verbindungen hat
        degree = graph.graph.degree(node_id)
        if degree == 0:
            isolated.append(node_id)
    
    return isolated


def link_isolated_nodes(
    isolated_node_ids: List[str],
    all_notes: Dict[str, AtomicNote],
    graph: GraphStore,
    llm_service: LLMService,
    similarity_threshold: float = 0.70,
    max_links_per_node: int = 3
) -> int:
    """
    Verlinkt isolierte Nodes automatisch mit ähnlichen Notes.
    
    Args:
        isolated_node_ids: Liste von isolierten Node-IDs
        all_notes: Dict aller Notes (note_id -> AtomicNote)
        graph: GraphStore Instanz
        llm_service: LLMService für Embedding-Berechnung
        similarity_threshold: Minimale Similarity für Verlinkung (default: 0.70, niedriger als suggest_relations)
            Wird dynamisch auf 0.50 reduziert, wenn gemeinsame Tags/Keywords vorhanden sind
        max_links_per_node: Maximale Anzahl Links pro isoliertem Node (default: 3)
    
    Returns:
        Anzahl erstellter Links
    
    Note:
        Die Funktion verwendet einen dynamischen Similarity-Threshold:
        - Standard: similarity_threshold (default: 0.70)
        - Bei gemeinsamen Tags/Keywords: 0.50 (reduziert um ~30%)
        - Gemeinsame Tags/Keywords sind ein starkes Signal für Relevanz und ermöglichen
          Verlinkung auch bei niedrigerer Embedding-Similarity
    """
    if not isolated_node_ids:
        return 0
    
    links_created = 0
    
    # Berechne Embeddings für alle Notes (einmalig)
    all_vectors = {}
    for note_id, note in all_notes.items():
        if note_id not in graph.graph.nodes:
            continue
        text = f"{note.content} {note.contextual_summary} {' '.join(note.keywords)}"
        try:
            embedding = llm_service.get_embedding(text)
            all_vectors[note_id] = embedding
        except Exception as e:
            log_debug(f"Error computing embedding for {note_id}: {e}")
            continue
    
    # Für jeden isolierten Node: Finde ähnliche Notes
    for isolated_id in isolated_node_ids:
        if isolated_id not in all_notes or isolated_id not in all_vectors:
            continue
        
        isolated_note = all_notes[isolated_id]
        isolated_vector = all_vectors[isolated_id]
        
        # Finde ähnliche Notes (nicht isoliert, nicht bereits verbunden)
        candidates = []
        for other_id, other_note in all_notes.items():
            if other_id == isolated_id:
                continue
            if other_id not in graph.graph.nodes:
                continue
            if other_id not in all_vectors:
                continue
            
            # Skip wenn bereits verbunden
            if graph.graph.has_edge(isolated_id, other_id) or graph.graph.has_edge(other_id, isolated_id):
                continue
            
            # Skip wenn auch isoliert (wir wollen isolierte mit verbundenen verlinken)
            if graph.graph.degree(other_id) == 0:
                continue
            
            # Berechne Similarity
            similarity = cosine_similarity(isolated_vector, all_vectors[other_id])
            
            # Pre-Filter: Gemeinsame Keywords/Tags erhöhen Wahrscheinlichkeit
            common_keywords = set(isolated_note.keywords) & set(other_note.keywords)
            common_tags = set(isolated_note.tags) & set(other_note.tags)
            has_common_attributes = bool(common_keywords or common_tags)
            
            # Dynamischer Threshold: Niedrigerer Threshold wenn gemeinsame Tags/Keywords vorhanden
            # Gemeinsame Tags/Keywords sind ein starkes Signal für Relevanz
            # Reduziere Threshold um 30% (von 0.70 auf 0.50) wenn gemeinsame Attribute vorhanden
            effective_threshold = 0.50 if has_common_attributes else similarity_threshold
            
            # Wenn Similarity hoch genug (mit dynamischem Threshold) ODER gemeinsame Keywords/Tags vorhanden
            if similarity >= effective_threshold or has_common_attributes:
                # Bonus für gemeinsame Keywords/Tags
                bonus = 0.1 if has_common_attributes else 0.0
                candidates.append((other_id, similarity + bonus, len(common_keywords) + len(common_tags)))
        
        # Sortiere Kandidaten nach Similarity + Bonus
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        # Erstelle Links zu den besten Kandidaten
        links_for_this_node = 0
        for other_id, similarity_score, common_count in candidates[:max_links_per_node]:
            if links_for_this_node >= max_links_per_node:
                break
            
            # Erstelle bidirektionale Relation
            relation_type = "related_to"
            if common_count > 0:
                relation_type = "similar_to"
            
            # Füge Edge hinzu
            if not graph.graph.has_edge(isolated_id, other_id):
                graph.graph.add_edge(
                    isolated_id,
                    other_id,
                    relation_type=relation_type,
                    weight=similarity_score,
                    created_at=datetime.utcnow().isoformat(),
                    source="enzyme_auto_link"
                )
                links_created += 1
                links_for_this_node += 1
                
                log_event("ISOLATED_NODE_LINKED", {
                    "isolated_id": isolated_id,
                    "linked_to": other_id,
                    "similarity": similarity_score,
                    "common_keywords_tags": common_count
                })
        
        if links_for_this_node > 0:
            log_event("ISOLATED_NODE_AUTO_LINKED", {
                "node_id": isolated_id,
                "links_created": links_for_this_node
            })
    
    if links_created > 0:
        log_event("ISOLATED_NODES_LINKED_TOTAL", {
            "count": links_created,
            "nodes_processed": len(isolated_node_ids)
        })
    
    return links_created


def _is_flag_too_old(flag_timestamp: Optional[str], max_age_days: int = 30) -> bool:
    """
    Prüft ob ein Flag zu alt ist.
    
    Args:
        flag_timestamp: ISO-Format Timestamp oder None
        max_age_days: Maximale Alter in Tagen (default: 30)
    
    Returns:
        True wenn Flag zu alt ist oder nicht existiert
    """
    if not flag_timestamp:
        return True
    
    try:
        flag_date = datetime.fromisoformat(flag_timestamp.replace('Z', '+00:00'))
        age = datetime.utcnow() - flag_date.replace(tzinfo=None)
        return age.days > max_age_days
    except Exception:
        return True  # Bei Parse-Fehler als zu alt behandeln


def normalize_and_clean_keywords(
    notes: Dict[str, AtomicNote],
    graph: GraphStore,
    llm_service: Optional[LLMService] = None,
    max_keywords: int = 7,
    max_flag_age_days: int = 30,
    ignore_flags: bool = False
) -> Dict[str, Any]:
    """
    Normalisiert und bereinigt Keywords:
    - Normalisiert Groß-/Kleinschreibung
    - Entfernt zu generische Keywords
    - Korrigiert falsche Keyword-Zuordnungen
    - Reduziert zu viele Keywords auf max_keywords
    
    Args:
        notes: Dict von note_id -> AtomicNote
        graph: GraphStore Instanz
        llm_service: Optional LLMService für Keyword-Korrektur
        max_keywords: Maximale Anzahl Keywords pro Note (default: 7)
    
    Returns:
        Dict mit Ergebnissen:
        - "keywords_normalized": Anzahl normalisierte Keywords
        - "keywords_removed": Anzahl entfernte Keywords
        - "keywords_corrected": Anzahl korrigierte Keywords
    """
    if len(notes) == 0:
        return {
            "keywords_normalized": 0,
            "keywords_removed": 0,
            "keywords_corrected": 0
        }
    
    normalized_count = 0
    removed_count = 0
    corrected_count = 0
    
    # Generische Keywords die entfernt/ersetzt werden sollten
    generic_keywords = {
        "befehl", "skript", "skripting", "tool", "reference", 
        "documentation", "guide", "tutorial", "how-to"
    }
    
    for node_id, note in notes.items():
        if node_id not in graph.graph.nodes:
            continue
        
        original_keywords = note.keywords.copy() if note.keywords else []
        new_keywords = []
        changed = False
        
        # 1. Normalisiere Keywords (lowercase, aber behalte wichtige Großschreibung)
        normalized_keywords = []
        for kw in original_keywords:
            kw_lower = kw.lower().strip()
            
            # Spezielle Fälle: Behalte Großschreibung für Akronyme
            if kw_lower in ["mcp", "api", "ide", "cli", "ui", "pdf", "json", "yaml", "html", "css", "js", "ts"]:
                normalized_kw = kw_lower.upper()
            elif kw_lower in ["javascript", "typescript", "python", "java", "c#", "go", "rust"]:
                # Programmiersprachen: Erste Buchstabe groß
                normalized_kw = kw_lower.capitalize()
            elif " " in kw_lower:
                # Mehrwort-Keywords: Title Case
                normalized_kw = " ".join(word.capitalize() for word in kw_lower.split())
            else:
                normalized_kw = kw_lower
            
            normalized_keywords.append(normalized_kw)
            if kw != normalized_kw:
                changed = True
        
        # 2. Entferne zu generische Keywords
        for kw in normalized_keywords:
            if kw.lower() not in generic_keywords:
                new_keywords.append(kw)
            else:
                removed_count += 1
                changed = True
        
        # 3. Prüfe auf falsche Keyword-Zuordnungen (via LLM)
        if llm_service and note.content and len(note.content.strip()) >= 50:
            # Prüfe ob Keywords zum Content passen
            keywords_str = ", ".join(new_keywords)
            prompt = f"""Prüfe ob diese Keywords zum Content passen:

Keywords: {keywords_str}

Content:
{note.content[:500]}

Antworte mit "JA" wenn alle Keywords passen, oder "NEIN" gefolgt von einer komma-separierten Liste mit 3-7 besseren Keywords die zum Content passen."""
            
            try:
                response = llm_service._call_llm(prompt)
                response_lower = response.strip().lower()
                
                if "nein" in response_lower or "no" in response_lower:
                    # Extrahiere neue Keywords
                    lines = response.split("\n")
                    new_keywords_text = None
                    for line in lines:
                        if "," in line or len(line.split()) <= 7:
                            new_keywords_text = line
                            break
                    
                    if new_keywords_text:
                        # Parse neue Keywords
                        keywords_text = new_keywords_text.strip().lower()
                        keywords_text = keywords_text.replace("nein", "").replace("no", "").strip()
                        keywords_text = keywords_text.replace("*", "").replace("-", "").replace("#", "")
                        corrected_keywords = [k.strip() for k in keywords_text.split(",") if k.strip()][:max_keywords]
                        
                        if corrected_keywords:
                            new_keywords = corrected_keywords
                            corrected_count += 1
                            changed = True
            except Exception as e:
                log_debug(f"Error correcting keywords for {node_id}: {e}")
        
        # 4. Reduziere auf max_keywords (behalte die wichtigsten)
        if len(new_keywords) > max_keywords:
            # Sortiere nach Länge (kürzere = spezifischer) und behalte die ersten max_keywords
            new_keywords = sorted(new_keywords, key=lambda x: (len(x), x))[:max_keywords]
            removed_count += len(original_keywords) - len(new_keywords)
            changed = True
        
        # 5. Entferne Duplikate
        seen = set()
        unique_keywords = []
        for kw in new_keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                seen.add(kw_lower)
                unique_keywords.append(kw)
            else:
                removed_count += 1
                changed = True
        
        # Update Graph
        if changed:
            note.keywords = unique_keywords
            graph.graph.nodes[node_id]["keywords"] = unique_keywords
            normalized_count += 1
            log_event("KEYWORDS_NORMALIZED", {
                "node_id": node_id,
                "old_keywords": original_keywords,
                "new_keywords": unique_keywords
            })
    
    if normalized_count > 0:
        log_event("KEYWORDS_NORMALIZED_TOTAL", {
            "normalized": normalized_count,
            "removed": removed_count,
            "corrected": corrected_count
        })
    
    return {
        "keywords_normalized": normalized_count,
        "keywords_removed": removed_count,
        "keywords_corrected": corrected_count
    }


def calculate_quality_score(
    note: AtomicNote,
    graph: GraphStore,
    node_id: str
) -> Dict[str, Any]:
    """
    Berechnet einen Quality-Score für eine Note (0.0 - 1.0).
    
    Bewertet:
    - Content-Qualität (Länge, Vollständigkeit)
    - Summary-Qualität (Länge, Spezifität)
    - Keywords/Tags-Qualität (Anzahl, Relevanz)
    - Verlinkungsgrad (Anzahl Connections)
    - Metadata-Vollständigkeit
    
    Args:
        note: AtomicNote
        graph: GraphStore Instanz
        node_id: ID der Note
    
    Returns:
        Dict mit:
        - "score": Gesamt-Score (0.0 - 1.0)
        - "content_score": Content-Qualität (0.0 - 1.0)
        - "summary_score": Summary-Qualität (0.0 - 1.0)
        - "keywords_score": Keywords-Qualität (0.0 - 1.0)
        - "tags_score": Tags-Qualität (0.0 - 1.0)
        - "linking_score": Verlinkungsgrad (0.0 - 1.0)
        - "metadata_score": Metadata-Vollständigkeit (0.0 - 1.0)
        - "issues": Liste von Quality-Issues
    """
    issues = []
    scores = {}
    
    # 1. Content-Score (0-25 Punkte)
    content_length = len(note.content.strip()) if note.content else 0
    if content_length == 0:
        content_score = 0.0
        issues.append("empty_content")
    elif content_length < 50:
        content_score = 0.3
        issues.append("content_too_short")
    elif content_length < 200:
        content_score = 0.6
        issues.append("content_short")
    elif content_length < 500:
        content_score = 0.8
    else:
        content_score = 1.0
    
    scores["content_score"] = content_score
    
    # 2. Summary-Score (0-20 Punkte)
    summary_length = len(note.contextual_summary.strip()) if note.contextual_summary else 0
    if summary_length == 0:
        summary_score = 0.0
        issues.append("missing_summary")
    elif summary_length < 20:
        summary_score = 0.4
        issues.append("summary_too_short")
    elif summary_length < 50:
        summary_score = 0.7
        issues.append("summary_short")
    elif summary_length > 150:
        summary_score = 0.9  # Zu lang kann auch schlecht sein
        issues.append("summary_too_long")
    else:
        summary_score = 1.0
    
    scores["summary_score"] = summary_score
    
    # 3. Keywords-Score (0-15 Punkte)
    keywords_count = len(note.keywords) if note.keywords else 0
    if keywords_count == 0:
        keywords_score = 0.0
        issues.append("no_keywords")
    elif keywords_count < 2:
        keywords_score = 0.4
        issues.append("too_few_keywords")
    elif keywords_count < 5:
        keywords_score = 0.8
    elif keywords_count <= 7:
        keywords_score = 1.0
    else:
        keywords_score = 0.9  # Zu viele Keywords
        issues.append("too_many_keywords")
    
    scores["keywords_score"] = keywords_score
    
    # 4. Tags-Score (0-10 Punkte)
    tags_count = len(note.tags) if note.tags else 0
    if tags_count == 0:
        tags_score = 0.0
        issues.append("no_tags")
    elif tags_count < 2:
        tags_score = 0.6
        issues.append("too_few_tags")
    elif tags_count <= 5:
        tags_score = 1.0
    else:
        tags_score = 0.8  # Zu viele Tags
        issues.append("too_many_tags")
    
    scores["tags_score"] = tags_score
    
    # 5. Linking-Score (0-20 Punkte)
    if node_id in graph.graph.nodes:
        degree = graph.graph.degree(node_id)
        if degree == 0:
            linking_score = 0.0
            issues.append("isolated_node")
        elif degree == 1:
            linking_score = 0.4
            issues.append("weakly_linked")
        elif degree < 3:
            linking_score = 0.7
            issues.append("moderately_linked")
        elif degree <= 10:
            linking_score = 1.0
        else:
            linking_score = 0.9  # Zu viele Links können auch schlecht sein
            issues.append("overlinked")
    else:
        linking_score = 0.0
        issues.append("node_not_in_graph")
    
    scores["linking_score"] = linking_score
    
    # 6. Metadata-Score (0-10 Punkte)
    metadata_completeness = 0
    if note.metadata:
        if "source" in note.metadata:
            metadata_completeness += 2
        if "source_url" in note.metadata:
            metadata_completeness += 2
        if "validated_at" in note.metadata:
            metadata_completeness += 2
        if "summary_validated_at" in note.metadata:
            metadata_completeness += 1
        if "keywords_validated_at" in note.metadata:
            metadata_completeness += 1
        if "tags_validated_at" in note.metadata:
            metadata_completeness += 1
        if len(note.metadata) > 3:
            metadata_completeness += 1
    else:
        issues.append("no_metadata")
    
    metadata_score = metadata_completeness / 10.0
    scores["metadata_score"] = metadata_score
    
    # Gesamt-Score (gewichteter Durchschnitt)
    total_score = (
        content_score * 0.25 +
        summary_score * 0.20 +
        keywords_score * 0.15 +
        tags_score * 0.10 +
        linking_score * 0.20 +
        metadata_score * 0.10
    )
    
    scores["score"] = total_score
    
    # Quality-Level bestimmen
    if total_score >= 0.9:
        quality_level = "excellent"
    elif total_score >= 0.75:
        quality_level = "good"
    elif total_score >= 0.6:
        quality_level = "fair"
    elif total_score >= 0.4:
        quality_level = "poor"
    else:
        quality_level = "very_poor"
    
    scores["quality_level"] = quality_level
    
    return {
        **scores,
        "issues": issues
    }


def validate_notes(
    notes: Dict[str, AtomicNote],
    graph: GraphStore,
    llm_service: Optional[LLMService] = None,
    max_flag_age_days: int = 30,
    ignore_flags: bool = False
) -> Dict[str, Any]:
    """
    Validiert Notes und korrigiert fehlende/ungültige Felder.
    
    Prüft:
    - Content vorhanden und nicht leer
    - Summary vorhanden, sinnvoll und passend zum Content
    - Keywords vorhanden (mindestens 2) und passend zum Content
    - Tags vorhanden (mindestens 1) und passend zum Content
    - Type korrekt gesetzt
    - Metadata vollständig
    
    Verwendet Flag-System:
    - summary_validated_at: Wann Summary zuletzt validiert wurde
    - keywords_validated_at: Wann Keywords zuletzt validiert wurden
    - tags_validated_at: Wann Tags zuletzt validiert wurden
    - validated_at: Wann Note zuletzt vollständig validiert wurde
    
    Args:
        notes: Dict von note_id -> AtomicNote
        graph: GraphStore Instanz
        llm_service: Optional LLMService für automatische Korrekturen
        max_flag_age_days: Maximale Alter der Flags in Tagen (default: 30)
    
    Returns:
        Dict mit Validierungs-Ergebnissen:
        - "validated": Anzahl validierte Notes
        - "corrected": Anzahl korrigierte Notes
        - "issues_found": Dict mit Issue-Typen und Anzahl
    """
    if len(notes) == 0:
        return {
            "validated": 0,
            "corrected": 0,
            "issues_found": {}
        }
    
    validated_count = 0
    corrected_count = 0
    issues = {
        "missing_summary": 0,
        "summary_mismatch": 0,  # Summary passt nicht zum Content
        "empty_keywords": 0,
        "keywords_mismatch": 0,  # Keywords passen nicht zum Content
        "empty_tags": 0,
        "tags_mismatch": 0,  # Tags passen nicht zum Content
        "invalid_type": 0,
        "short_content": 0,
        "missing_metadata": 0
    }
    
    valid_types = ["rule", "procedure", "concept", "tool", "reference", "integration"]
    now_iso = datetime.utcnow().isoformat()
    
    for node_id, note in notes.items():
        if node_id not in graph.graph.nodes:
            continue
        
        # Initialisiere Metadata falls nicht vorhanden
        if not note.metadata:
            note.metadata = {}
            graph.graph.nodes[node_id]["metadata"] = {}
        
        metadata = note.metadata
        needs_correction = False
        node_issues = []
        
        # Prüfe Content
        if not note.content or len(note.content.strip()) < 20:
            issues["short_content"] += 1
            node_issues.append("short_content")
            needs_correction = True
        
        # Prüfe Summary (mit Flag-Check)
        # Handle None metadata gracefully
        if metadata is None:
            metadata = {}
        summary_flag = metadata.get("summary_validated_at")
        # CRITICAL: Wenn Summary fehlt/leer ist, IMMER erstellen (unabhängig von Flags)
        summary_missing = not note.contextual_summary or len(note.contextual_summary.strip()) < 10
        should_validate_summary = summary_missing or ignore_flags or _is_flag_too_old(summary_flag, max_flag_age_days)
        
        if should_validate_summary:
            if summary_missing:
                issues["missing_summary"] += 1
                node_issues.append("missing_summary")
                needs_correction = True
                
                # Auto-Korrektur: Erstelle Summary aus Content
                if llm_service and note.content and len(note.content.strip()) >= 20:
                    try:
                        prompt = f"Erstelle eine kurze, prägnante Zusammenfassung (max 100 Zeichen) für diesen Content:\n\n{note.content[:500]}"
                        new_summary = llm_service._call_llm(prompt)
                        new_summary = new_summary.strip()[:100]
                        
                        if new_summary:
                            note.contextual_summary = new_summary
                            graph.graph.nodes[node_id]["contextual_summary"] = new_summary
                            metadata["summary_validated_at"] = now_iso
                            graph.graph.nodes[node_id]["metadata"] = metadata
                            corrected_count += 1
                            log_event("NOTE_SUMMARY_AUTO_CORRECTED", {
                                "node_id": node_id,
                                "new_summary": new_summary[:50]
                            })
                    except Exception as e:
                        log_debug(f"Error auto-correcting summary for {node_id}: {e}")
            else:
                # Prüfe ob Summary zum Content passt (via LLM)
                if llm_service and note.content and len(note.content.strip()) >= 50:
                    try:
                        prompt = f"""Prüfe ob dieser Summary zum Content passt:

Summary: {note.contextual_summary}

Content:
{note.content[:500]}

Antworte nur mit "JA" wenn der Summary zum Content passt, oder "NEIN" wenn nicht.
Wenn NEIN, gib einen besseren Summary (max 100 Zeichen) zurück."""
                        response = llm_service._call_llm(prompt)
                        response_lower = response.strip().lower()
                        
                        if "nein" in response_lower or "no" in response_lower or not response_lower.startswith("ja"):
                            # Extrahiere neuen Summary aus Response
                            lines = response.split("\n")
                            new_summary = None
                            for line in lines:
                                if len(line.strip()) > 10 and len(line.strip()) <= 100:
                                    if "summary" not in line.lower() or len([l for l in lines if l.strip()]) == 1:
                                        new_summary = line.strip()[:100]
                                        break
                            
                            if not new_summary:
                                # Fallback: Erstelle neuen Summary
                                prompt2 = f"Erstelle eine kurze, prägnante Zusammenfassung (max 100 Zeichen) für diesen Content:\n\n{note.content[:500]}"
                                new_summary = llm_service._call_llm(prompt2).strip()[:100]
                            
                            if new_summary:
                                issues["summary_mismatch"] += 1
                                node_issues.append("summary_mismatch")
                                note.contextual_summary = new_summary
                                graph.graph.nodes[node_id]["contextual_summary"] = new_summary
                                metadata["summary_validated_at"] = now_iso
                                graph.graph.nodes[node_id]["metadata"] = metadata
                                corrected_count += 1
                                log_event("NOTE_SUMMARY_CORRECTED_MISMATCH", {
                                    "node_id": node_id,
                                    "old_summary": note.contextual_summary[:50],
                                    "new_summary": new_summary[:50]
                                })
                        else:
                            # Summary passt, setze Flag
                            metadata["summary_validated_at"] = now_iso
                            graph.graph.nodes[node_id]["metadata"] = metadata
                    except Exception as e:
                        log_debug(f"Error validating summary for {node_id}: {e}")
                else:
                    # Kein LLM, setze Flag trotzdem
                    metadata["summary_validated_at"] = now_iso
                    graph.graph.nodes[node_id]["metadata"] = metadata
        
        # Prüfe Keywords (mit Flag-Check)
        # Handle None metadata gracefully (metadata might have been reassigned)
        if metadata is None:
            metadata = {}
        keywords_flag = metadata.get("keywords_validated_at")
        # CRITICAL: Wenn Keywords fehlen/leer sind, IMMER erstellen (unabhängig von Flags)
        keywords_missing = not note.keywords or len(note.keywords) < 2
        should_validate_keywords = keywords_missing or ignore_flags or _is_flag_too_old(keywords_flag, max_flag_age_days)
        
        if should_validate_keywords:
            if keywords_missing:
                issues["empty_keywords"] += 1
                node_issues.append("empty_keywords")
                needs_correction = True
                
                # Auto-Korrektur: Extrahiere Keywords aus Content
                if llm_service and note.content and len(note.content.strip()) >= 50:
                    try:
                        prompt = f"Extrahiere 3-5 wichtige Keywords aus diesem Text:\n\n{note.content[:500]}\n\nGib nur die Keywords als komma-separierte Liste zurück, keine Erklärung."
                        response = llm_service._call_llm(prompt)
                        keywords_text = response.strip().lower()
                        keywords_text = keywords_text.replace("*", "").replace("-", "").replace("#", "")
                        new_keywords = [k.strip() for k in keywords_text.split(",") if k.strip()][:5]
                        
                        if new_keywords:
                            note.keywords = new_keywords
                            graph.graph.nodes[node_id]["keywords"] = new_keywords
                            metadata["keywords_validated_at"] = now_iso
                            graph.graph.nodes[node_id]["metadata"] = metadata
                            corrected_count += 1
                            log_event("NOTE_KEYWORDS_AUTO_CORRECTED", {
                                "node_id": node_id,
                                "new_keywords": new_keywords
                            })
                    except Exception as e:
                        log_debug(f"Error auto-correcting keywords for {node_id}: {e}")
            else:
                # Prüfe ob Keywords zum Content passen
                if llm_service and note.content and len(note.content.strip()) >= 50:
                    try:
                        keywords_str = ", ".join(note.keywords)
                        prompt = f"""Prüfe ob diese Keywords zum Content passen:

Keywords: {keywords_str}

Content:
{note.content[:500]}

Antworte nur mit "JA" wenn die Keywords zum Content passen, oder "NEIN" wenn nicht.
Wenn NEIN, gib eine komma-separierte Liste mit 3-5 besseren Keywords zurück."""
                        response = llm_service._call_llm(prompt)
                        response_lower = response.strip().lower()
                        
                        if "nein" in response_lower or "no" in response_lower or not response_lower.startswith("ja"):
                            # Extrahiere neue Keywords aus Response
                            lines = response.split("\n")
                            new_keywords_text = None
                            for line in lines:
                                if "," in line or len(line.split()) <= 5:
                                    new_keywords_text = line
                                    break
                            
                            if not new_keywords_text:
                                # Fallback: Extrahiere Keywords neu
                                prompt2 = f"Extrahiere 3-5 wichtige Keywords aus diesem Text:\n\n{note.content[:500]}\n\nGib nur die Keywords als komma-separierte Liste zurück."
                                new_keywords_text = llm_service._call_llm(prompt2)
                            
                            keywords_text = new_keywords_text.strip().lower()
                            keywords_text = keywords_text.replace("*", "").replace("-", "").replace("#", "")
                            new_keywords = [k.strip() for k in keywords_text.split(",") if k.strip()][:5]
                            
                            if new_keywords:
                                issues["keywords_mismatch"] += 1
                                node_issues.append("keywords_mismatch")
                                note.keywords = new_keywords
                                graph.graph.nodes[node_id]["keywords"] = new_keywords
                                metadata["keywords_validated_at"] = now_iso
                                graph.graph.nodes[node_id]["metadata"] = metadata
                                corrected_count += 1
                                log_event("NOTE_KEYWORDS_CORRECTED_MISMATCH", {
                                    "node_id": node_id,
                                    "old_keywords": note.keywords,
                                    "new_keywords": new_keywords
                                })
                        else:
                            # Keywords passen, setze Flag
                            metadata["keywords_validated_at"] = now_iso
                            graph.graph.nodes[node_id]["metadata"] = metadata
                    except Exception as e:
                        log_debug(f"Error validating keywords for {node_id}: {e}")
                else:
                    # Kein LLM, setze Flag trotzdem
                    metadata["keywords_validated_at"] = now_iso
                    graph.graph.nodes[node_id]["metadata"] = metadata
        
        # Prüfe Type (mit Flag-Check)
        type_flag = metadata.get("type_validated_at")
        should_validate_type = not note.type or ignore_flags or _is_flag_too_old(type_flag, max_flag_age_days)
        
        if should_validate_type:
            if not note.type or note.type not in valid_types:
                issues["invalid_type"] += 1
                node_issues.append("invalid_type")
                needs_correction = True
                
                # Auto-Korrektur: Bestimme Type basierend auf Content
                if llm_service and note.content and len(note.content.strip()) >= 50:
                    try:
                        prompt = f"""Bestimme den Type dieser Note basierend auf dem Content. Mögliche Types: rule, procedure, concept, tool, reference, integration.

Content:
{note.content[:500]}

Antworte nur mit einem der Types: rule, procedure, concept, tool, reference, integration"""
                        response = llm_service._call_llm(prompt)
                        response_lower = response.strip().lower()
                        
                        # Extrahiere Type aus Response
                        detected_type = None
                        for valid_type in valid_types:
                            if valid_type in response_lower:
                                detected_type = valid_type
                                break
                        
                        if not detected_type:
                            # Fallback: Bestimme Type basierend auf Keywords/Content
                            content_lower = note.content.lower()
                            if any(kw in content_lower for kw in ["how to", "anleitung", "schritt", "tutorial", "setup"]):
                                detected_type = "procedure"
                            elif any(kw in content_lower for kw in ["tool", "werkzeug", "utility", "script"]):
                                detected_type = "tool"
                            elif any(kw in content_lower for kw in ["regel", "rule", "guideline", "best practice"]):
                                detected_type = "rule"
                            elif any(kw in content_lower for kw in ["konzept", "concept", "theorie", "theory"]):
                                detected_type = "concept"
                            elif any(kw in content_lower for kw in ["referenz", "reference", "dokumentation", "docs"]):
                                detected_type = "reference"
                            else:
                                detected_type = "concept"  # Default
                        
                        if detected_type:
                            note.type = detected_type
                            graph.graph.nodes[node_id]["type"] = detected_type
                            metadata["type_validated_at"] = now_iso
                            graph.graph.nodes[node_id]["metadata"] = metadata
                            corrected_count += 1
                            log_event("NOTE_TYPE_AUTO_CORRECTED", {
                                "node_id": node_id,
                                "old_type": note.type,
                                "new_type": detected_type
                            })
                    except Exception as e:
                        log_debug(f"Error auto-correcting type for {node_id}: {e}")
                        # Fallback: Setze Default-Type
                        if not note.type:
                            note.type = "concept"
                            graph.graph.nodes[node_id]["type"] = "concept"
                            metadata["type_validated_at"] = now_iso
                            graph.graph.nodes[node_id]["metadata"] = metadata
                else:
                    # Kein LLM, setze Default-Type
                    if not note.type:
                        note.type = "concept"
                        graph.graph.nodes[node_id]["type"] = "concept"
                        metadata["type_validated_at"] = now_iso
                        graph.graph.nodes[node_id]["metadata"] = metadata
            else:
                # Type ist gültig, setze Flag
                metadata["type_validated_at"] = now_iso
                graph.graph.nodes[node_id]["metadata"] = metadata
        
        # Prüfe Tags (mit Flag-Check)
        # Handle None metadata gracefully (metadata might have been reassigned)
        if metadata is None:
            metadata = {}
        tags_flag = metadata.get("tags_validated_at")
        # CRITICAL: Wenn Tags fehlen/leer sind, IMMER erstellen (unabhängig von Flags)
        tags_missing = not note.tags or len(note.tags) < 1
        should_validate_tags = tags_missing or ignore_flags or _is_flag_too_old(tags_flag, max_flag_age_days)
        
        if should_validate_tags:
            if tags_missing:
                issues["empty_tags"] += 1
                node_issues.append("empty_tags")
                needs_correction = True
                
                # Auto-Korrektur: Erstelle Tags basierend auf Type und Content
                if note.type:
                    type_to_tags = {
                        "tool": ["tool", "reference"],
                        "procedure": ["procedure", "how-to"],
                        "concept": ["concept", "theory"],
                        "reference": ["reference", "documentation"],
                        "integration": ["integration", "setup"],
                        "rule": ["rule", "guideline"]
                    }
                    default_tags = type_to_tags.get(note.type, ["general"])
                    note.tags = default_tags
                    graph.graph.nodes[node_id]["tags"] = default_tags
                    metadata["tags_validated_at"] = now_iso
                    graph.graph.nodes[node_id]["metadata"] = metadata
                    corrected_count += 1
                    log_event("NOTE_TAGS_AUTO_CORRECTED", {
                        "node_id": node_id,
                        "new_tags": default_tags
                    })
            elif len(note.tags) > 5:
                # Zu viele Tags: Reduziere auf die wichtigsten 4-5
                issues["too_many_tags"] = issues.get("too_many_tags", 0) + 1
                node_issues.append("too_many_tags")
                needs_correction = True
                
                # Reduziere Tags: Behalte die ersten 4-5 (wichtigste)
                # Oder nutze LLM für intelligente Reduktion
                if llm_service and note.content and len(note.content.strip()) >= 50:
                    try:
                        tags_str = ", ".join(note.tags)
                        prompt = f"""Diese Note hat zu viele Tags ({len(note.tags)}). Reduziere auf die 4-5 wichtigsten Tags:

Aktuelle Tags: {tags_str}

Content:
{note.content[:500]}

Gib nur die 4-5 wichtigsten Tags als komma-separierte Liste zurück, keine Erklärung."""
                        response = llm_service._call_llm(prompt)
                        tags_text = response.strip().lower()
                        tags_text = tags_text.replace("*", "").replace("-", "").replace("#", "")
                        new_tags = [t.strip() for t in tags_text.split(",") if t.strip()][:5]
                        
                        if new_tags and len(new_tags) <= 5:
                            note.tags = new_tags
                            graph.graph.nodes[node_id]["tags"] = new_tags
                            metadata["tags_validated_at"] = now_iso
                            graph.graph.nodes[node_id]["metadata"] = metadata
                            corrected_count += 1
                            log_event("NOTE_TAGS_REDUCED", {
                                "node_id": node_id,
                                "old_tags": note.tags,
                                "new_tags": new_tags
                            })
                    except Exception as e:
                        log_debug(f"Error reducing tags for {node_id}: {e}")
                        # Fallback: Einfach die ersten 5 behalten
                        note.tags = note.tags[:5]
                        graph.graph.nodes[node_id]["tags"] = note.tags
                        metadata["tags_validated_at"] = now_iso
                        graph.graph.nodes[node_id]["metadata"] = metadata
                        corrected_count += 1
                        log_event("NOTE_TAGS_REDUCED_FALLBACK", {
                            "node_id": node_id,
                            "new_tags": note.tags
                        })
                else:
                    # Kein LLM: Einfach die ersten 5 behalten
                    note.tags = note.tags[:5]
                    graph.graph.nodes[node_id]["tags"] = note.tags
                    metadata["tags_validated_at"] = now_iso
                    graph.graph.nodes[node_id]["metadata"] = metadata
                    corrected_count += 1
                    log_event("NOTE_TAGS_REDUCED_FALLBACK", {
                        "node_id": node_id,
                        "new_tags": note.tags
                    })
            else:
                # Prüfe ob Tags zum Content passen
                if llm_service and note.content and len(note.content.strip()) >= 50:
                    try:
                        tags_str = ", ".join(note.tags)
                        prompt = f"""Prüfe ob diese Tags zum Content passen:

Tags: {tags_str}

Content:
{note.content[:500]}

Antworte nur mit "JA" wenn die Tags zum Content passen, oder "NEIN" wenn nicht.
Wenn NEIN, gib eine komma-separierte Liste mit 2-3 besseren Tags zurück."""
                        response = llm_service._call_llm(prompt)
                        response_lower = response.strip().lower()
                        
                        if "nein" in response_lower or "no" in response_lower or not response_lower.startswith("ja"):
                            # Extrahiere neue Tags aus Response
                            lines = response.split("\n")
                            new_tags_text = None
                            for line in lines:
                                if "," in line or len(line.split()) <= 3:
                                    new_tags_text = line
                                    break
                            
                            if not new_tags_text:
                                # Fallback: Erstelle Tags basierend auf Type
                                type_to_tags = {
                                    "tool": ["tool", "reference"],
                                    "procedure": ["procedure", "how-to"],
                                    "concept": ["concept", "theory"],
                                    "reference": ["reference", "documentation"],
                                    "integration": ["integration", "setup"],
                                    "rule": ["rule", "guideline"]
                                }
                                new_tags = type_to_tags.get(note.type, ["general"])
                            else:
                                tags_text = new_tags_text.strip().lower()
                                tags_text = tags_text.replace("*", "").replace("-", "").replace("#", "")
                                new_tags = [t.strip() for t in tags_text.split(",") if t.strip()][:3]
                            
                            if new_tags:
                                issues["tags_mismatch"] += 1
                                node_issues.append("tags_mismatch")
                                note.tags = new_tags
                                graph.graph.nodes[node_id]["tags"] = new_tags
                                metadata["tags_validated_at"] = now_iso
                                graph.graph.nodes[node_id]["metadata"] = metadata
                                corrected_count += 1
                                log_event("NOTE_TAGS_CORRECTED_MISMATCH", {
                                    "node_id": node_id,
                                    "old_tags": note.tags,
                                    "new_tags": new_tags
                                })
                        else:
                            # Tags passen, setze Flag
                            metadata["tags_validated_at"] = now_iso
                            graph.graph.nodes[node_id]["metadata"] = metadata
                    except Exception as e:
                        log_debug(f"Error validating tags for {node_id}: {e}")
                else:
                    # Kein LLM, setze Flag trotzdem
                    metadata["tags_validated_at"] = now_iso
                    graph.graph.nodes[node_id]["metadata"] = metadata
        
        # Type-Validierung wird jetzt in validate_note_types() gemacht
        # Hier nur noch prüfen ob Type vorhanden ist für Quality-Score
        
        # Berechne Quality-Score
        quality_data = calculate_quality_score(note, graph, node_id)
        metadata["quality_score"] = quality_data["score"]
        metadata["quality_level"] = quality_data["quality_level"]
        metadata["quality_scores"] = {
            "content": quality_data["content_score"],
            "summary": quality_data["summary_score"],
            "keywords": quality_data["keywords_score"],
            "tags": quality_data["tags_score"],
            "linking": quality_data["linking_score"],
            "metadata": quality_data["metadata_score"]
        }
        metadata["quality_issues"] = quality_data["issues"]
        metadata["quality_calculated_at"] = now_iso
        
        # Setze validated_at Flag
        metadata["validated_at"] = now_iso
        graph.graph.nodes[node_id]["metadata"] = metadata
        
        validated_count += 1
        
        if node_issues:
            log_event("NOTE_VALIDATION_ISSUES", {
                "node_id": node_id,
                "issues": node_issues
            })
    
    if validated_count > 0:
        log_event("NOTES_VALIDATED", {
            "validated": validated_count,
            "corrected": corrected_count,
            "issues": issues
        })
    
    return {
        "validated": validated_count,
        "corrected": corrected_count,
        "issues_found": issues
    }


def remove_low_quality_notes(
    notes: Dict[str, AtomicNote],
    graph: GraphStore,
    llm_service: Optional[LLMService] = None
) -> int:
    """
    Entfernt Notes mit irrelevantem oder fehlerhaftem Content.
    
    Erkennt:
    - CAPTCHA-Content
    - Fehlerseiten/Redirect-Seiten
    - Leere/ungültige Content
    - Irrelevante Content-Patterns
    
    Args:
        notes: Dict von note_id -> AtomicNote
        graph: GraphStore Instanz
        llm_service: Optional LLMService für Content-Validierung (wenn None, nur Pattern-Matching)
    
    Returns:
        Anzahl entfernte Notes
    """
    if len(notes) == 0:
        return 0
    
    to_remove = []
    
    # Pattern-basierte Erkennung von irrelevantem Content
    irrelevant_patterns = [
        r"CAPTCHA",
        r"captcha",
        r"human verification",
        r"verify you.*human",
        r"security challenge",
        r"access denied",
        r"403 forbidden",
        r"404 not found",
        r"page not found",
        r"redirect",
        r"please enable javascript",
        r"cookie consent",
        r"privacy policy",
        r"terms of service",
        r"subscribe to newsletter"
    ]
    
    import re
    
    for node_id, note in notes.items():
        if node_id not in graph.graph.nodes:
            continue
        
        content = note.content.strip().lower()
        should_remove = False
        reason = ""
        
        # Prüfe auf irrelevante Patterns
        for pattern in irrelevant_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                should_remove = True
                reason = f"matches_irrelevant_pattern: {pattern}"
                break
        
        # Prüfe auf sehr kurzen/leeren Content (nach Strip)
        if len(content) < 50:
            should_remove = True
            reason = "content_too_short"
        
        # Prüfe auf Source-URL-Probleme (wenn metadata vorhanden)
        if note.metadata:
            source_url = note.metadata.get("source_url", "")
            if source_url:
                # Prüfe auf bekannte Fehler-URLs
                error_url_patterns = [
                    r"captcha",
                    r"error",
                    r"403",
                    r"404",
                    r"blocked",
                    r"denied"
                ]
                for url_pattern in error_url_patterns:
                    if re.search(url_pattern, source_url, re.IGNORECASE):
                        should_remove = True
                        reason = f"error_url_pattern: {url_pattern}"
                        break
        
        # Prüfe auf Researcher-Agent Notes mit problematischem Content
        if note.metadata and note.metadata.get("source") == "researcher_agent":
            # Wenn Content sehr generisch oder CAPTCHA-ähnlich
            if "captcha" in content or "verification" in content.lower():
                # Zusätzliche Prüfung: Ist es wirklich relevant?
                if "zettelkasten" not in content.lower() and "knowledge graph" not in content.lower():
                    should_remove = True
                    reason = "researcher_captcha_content"
        
        if should_remove:
            to_remove.append((node_id, reason))
    
    # Entferne Notes
    for node_id, reason in to_remove:
        if node_id in graph.graph.nodes:
            graph.graph.remove_node(node_id)
            log_event("LOW_QUALITY_NOTE_REMOVED", {
                "node_id": node_id,
                "reason": reason,
                "summary": notes[node_id].contextual_summary[:50] if node_id in notes else "N/A"
            })
    
    if to_remove:
        log_event("LOW_QUALITY_NOTES_REMOVED", {
            "count": len(to_remove),
            "reasons": {r: sum(1 for _, reason in to_remove if reason == r) for _, r in to_remove}
        })
    
    return len(to_remove)


def refine_summaries(
    notes: Dict[str, AtomicNote],
    graph: GraphStore,
    llm_service: LLMService,
    similarity_threshold: float = 0.75,  # Niedrigerer Threshold für bessere Erkennung
    max_refinements: int = 10
) -> int:
    """
    Findet Notes mit ähnlichen Summarys und macht sie spezifischer.
    
    Args:
        notes: Dict von note_id -> AtomicNote
        graph: GraphStore Instanz
        llm_service: LLMService für Embedding-Berechnung und LLM-Calls
        similarity_threshold: Minimale Summary-Similarity für Refinement (default: 0.85)
        max_refinements: Maximale Anzahl Refinements pro Run (default: 10)
    
    Returns:
        Anzahl verfeinerte Summarys
    """
    if len(notes) < 2:
        return 0
    
    refined_count = 0
    note_ids = list(notes.keys())
    
    # Berechne Embeddings für alle Summarys
    summary_embeddings = {}
    for note_id, note in notes.items():
        if note.contextual_summary:
            embedding = llm_service.get_embedding(note.contextual_summary)
            summary_embeddings[note_id] = embedding
    
    # Finde Paare mit ähnlichen Summarys
    similar_pairs = []
    for i in range(len(note_ids)):
        if refined_count >= max_refinements:
            break
            
        for j in range(i + 1, len(note_ids)):
            if refined_count >= max_refinements:
                break
            
            a_id, b_id = note_ids[i], note_ids[j]
            
            # Prüfe ob beide Summarys haben
            if a_id not in summary_embeddings or b_id not in summary_embeddings:
                continue
            
            note_a = notes[a_id]
            note_b = notes[b_id]
            
            # Skip wenn bereits verfeinert (via metadata flag)
            if note_a.metadata.get("summary_refined", False) or note_b.metadata.get("summary_refined", False):
                continue  # Bereits verfeinert, skip
            
            # Prüfe ob Summarys identisch oder sehr ähnlich sind
            summary_a = note_a.contextual_summary.strip()
            summary_b = note_b.contextual_summary.strip()
            
            # Exakte Übereinstimmung
            if summary_a == summary_b and summary_a:
                similar_pairs.append((a_id, b_id, 1.0))
                continue
            
            # Embedding-Similarity
            similarity = cosine_similarity(
                summary_embeddings[a_id],
                summary_embeddings[b_id]
            )
            
            if similarity >= similarity_threshold:
                similar_pairs.append((a_id, b_id, similarity))
    
    # Verfeinere Summarys für ähnliche Paare
    for a_id, b_id, similarity in similar_pairs[:max_refinements]:
        if refined_count >= max_refinements:
            break
        
        note_a = notes[a_id]
        note_b = notes[b_id]
        
        # Prüfe ob Content unterschiedlich ist (sonst macht Refinement keinen Sinn)
        if note_a.content == note_b.content:
            continue  # Gleicher Content = gleicher Summary ist OK
        
        # Erstelle spezifischere Summarys via LLM
        try:
            # Refine Note A
            prompt_a = f"""Erstelle einen spezifischeren, eindeutigen Summary für diese Note.
Der aktuelle Summary ist zu generisch und wird auch von anderen Notes verwendet.

Aktueller Summary: {note_a.contextual_summary}

Note Content:
{note_a.content[:500]}

Keywords: {', '.join(note_a.keywords[:5])}
Tags: {', '.join(note_a.tags[:3])}

Erstelle einen neuen Summary (max 100 Zeichen) der:
1. Spezifisch für DIESE Note ist
2. Den einzigartigen Aspekt hervorhebt
3. Von anderen Notes unterscheidbar ist
4. Prägnant und informativ bleibt

Neuer Summary:"""
            
            new_summary_a = llm_service._call_llm(prompt_a).strip()
            
            # Refine Note B
            prompt_b = f"""Erstelle einen spezifischeren, eindeutigen Summary für diese Note.
Der aktuelle Summary ist zu generisch und wird auch von anderen Notes verwendet.

Aktueller Summary: {note_b.contextual_summary}

Note Content:
{note_b.content[:500]}

Keywords: {', '.join(note_b.keywords[:5])}
Tags: {', '.join(note_b.tags[:3])}

Erstelle einen neuen Summary (max 100 Zeichen) der:
1. Spezifisch für DIESE Note ist
2. Den einzigartigen Aspekt hervorhebt
3. Von anderen Notes unterscheidbar ist
4. Prägnant und informativ bleibt

Neuer Summary:"""
            
            new_summary_b = llm_service._call_llm(prompt_b).strip()
            
            # Update Graph Nodes
            if new_summary_a and new_summary_a != note_a.contextual_summary:
                if a_id in graph.graph.nodes:
                    graph.graph.nodes[a_id]["contextual_summary"] = new_summary_a
                    # Markiere als verfeinert (verhindert erneute Verarbeitung)
                    if "metadata" not in graph.graph.nodes[a_id]:
                        graph.graph.nodes[a_id]["metadata"] = {}
                    graph.graph.nodes[a_id]["metadata"]["summary_refined"] = True
                    graph.graph.nodes[a_id]["metadata"]["summary_refined_at"] = datetime.utcnow().isoformat()
                    refined_count += 1
                    log_event("SUMMARY_REFINED", {
                        "note_id": a_id,
                        "old_summary": note_a.contextual_summary[:50],
                        "new_summary": new_summary_a[:50]
                    })
            
            if new_summary_b and new_summary_b != note_b.contextual_summary:
                if b_id in graph.graph.nodes:
                    graph.graph.nodes[b_id]["contextual_summary"] = new_summary_b
                    # Markiere als verfeinert (verhindert erneute Verarbeitung)
                    if "metadata" not in graph.graph.nodes[b_id]:
                        graph.graph.nodes[b_id]["metadata"] = {}
                    graph.graph.nodes[b_id]["metadata"]["summary_refined"] = True
                    graph.graph.nodes[b_id]["metadata"]["summary_refined_at"] = datetime.utcnow().isoformat()
                    refined_count += 1
                    log_event("SUMMARY_REFINED", {
                        "note_id": b_id,
                        "old_summary": note_b.contextual_summary[:50],
                        "new_summary": new_summary_b[:50]
                    })
        
        except Exception as e:
            log_debug(f"Error refining summaries for {a_id}/{b_id}: {e}")
            continue
    
    if refined_count > 0:
        log_event("SUMMARIES_REFINED", {
            "count": refined_count,
            "similarity_threshold": similarity_threshold
        })
    
    return refined_count


def suggest_relations(
    notes: Dict[str, AtomicNote],
    graph: GraphStore,
    llm_service: LLMService,
    threshold: float = 0.75,
    max_suggestions: int = 10
) -> List[Tuple[str, str, float]]:
    """
    Schlägt neue Beziehungen zwischen Notes vor basierend auf semantischer Ähnlichkeit.
    
    Args:
        notes: Dict von note_id -> AtomicNote
        llm_service: LLMService für Embedding-Berechnung
        threshold: Minimale Similarity (default: 0.75)
        max_suggestions: Maximale Anzahl Vorschläge (default: 10)
    
    Returns:
        Liste von (source_id, target_id, similarity) Tupeln
    """
    if len(notes) < 2:
        return []
    
    suggestions = []
    note_ids = list(notes.keys())
    
    # Embeddings einmal berechnen
    vectors = {}
    for note_id, note in notes.items():
        text = f"{note.content} {note.contextual_summary} {' '.join(note.keywords)}"
        embedding = llm_service.get_embedding(text)
        vectors[note_id] = embedding
    
    # Paarweiser Vergleich (nur wenn nicht bereits verbunden)
    for i in range(len(note_ids)):
        for j in range(i + 1, len(note_ids)):
            if len(suggestions) >= max_suggestions:
                break
            
            a_id, b_id = note_ids[i], note_ids[j]
            
            note_a = notes[a_id]
            note_b = notes[b_id]
            
            # Prüfe ob bereits verbunden
            if graph.graph.has_edge(a_id, b_id) or graph.graph.has_edge(b_id, a_id):
                continue  # Bereits verbunden, Skip
            
            # Pre-Filter: Wenn keine gemeinsamen Keywords/Tags → Skip
            common_keywords = set(note_a.keywords) & set(note_b.keywords)
            common_tags = set(note_a.tags) & set(note_b.tags)
            
            if not common_keywords and not common_tags:
                continue  # Zu unterschiedlich, Skip
            
            # Cosine Similarity berechnen
            similarity = cosine_similarity(vectors[a_id], vectors[b_id])
            
            if similarity >= threshold:
                suggestions.append((a_id, b_id, similarity))
    
    # Sortiere nach Similarity (höchste zuerst)
    suggestions.sort(key=lambda x: x[2], reverse=True)
    
    return suggestions[:max_suggestions]


def digest_node(
    node_id: str,
    child_notes: List[AtomicNote],
    llm_service: LLMService,
    max_children: int = 8
) -> Optional[str]:
    """
    Wenn ein Node zu viele Kinder hat, erzeugt eine kompakte Zusammenfassung.
    
    Args:
        node_id: ID des überfüllten Nodes
        child_notes: Liste der Child-Notes
        llm_service: LLMService für Zusammenfassung
        max_children: Maximale Anzahl Children bevor Digest nötig ist
    
    Returns:
        Zusammenfassungstext oder None wenn nicht nötig
    """
    if len(child_notes) <= max_children:
        return None
    
    # Sammle Content aller Children
    texts = "\n\n---\n\n".join([
        f"[{note.id}] {note.content}\nSummary: {note.contextual_summary}\nKeywords: {', '.join(note.keywords)}"
        for note in child_notes
    ])
    
    prompt = f"""Fasse folgende {len(child_notes)} Notizen prägnant zusammen.
Ziel: Eine abstrahierte, verdichtete Meta-Note die die Essenz aller Notizen erfasst.

Notizen:
{texts}

Erstelle eine kompakte Zusammenfassung (max 200 Wörter) die:
1. Die Hauptthemen zusammenfasst
2. Gemeinsame Patterns identifiziert
3. Wichtige Details bewahrt
4. Redundanzen eliminiert

Zusammenfassung:"""
    
    try:
        summary = llm_service._call_llm(prompt)
        log_event("NODE_DIGESTED", {
            "node_id": node_id,
            "children_count": len(child_notes),
            "summary_length": len(summary)
        })
        return summary
    except Exception as e:
        log_debug(f"Digest Error für Node {node_id}: {e}")
        return None


def repair_corrupted_nodes(graph: GraphStore) -> int:
    """
    Repariert korrupte Nodes (z.B. created_at = 'None' String statt datetime).
    
    Args:
        graph: GraphStore Instanz
    
    Returns:
        Anzahl reparierte Nodes
    """
    repaired_count = 0
    
    for node_id in list(graph.graph.nodes()):
        try:
            node_data = graph.graph.nodes[node_id]
            # Handle None node_data gracefully
            if node_data is None:
                node_data = {}
            
            needs_repair = False
            repairs = {}
            
            # Check for corrupted created_at (string 'None' or empty string)
            if 'created_at' in node_data:
                created_at_val = node_data['created_at']
                if isinstance(created_at_val, str):
                    # Handle 'None' string case
                    if created_at_val.lower() == 'none' or created_at_val == '':
                        needs_repair = True
                        repairs['created_at'] = datetime.utcnow().isoformat()
                    else:
                        # Try to parse as ISO string
                        try:
                            datetime.fromisoformat(created_at_val)
                        except:
                            # Invalid format - repair it
                            needs_repair = True
                            repairs['created_at'] = datetime.utcnow().isoformat()
                elif created_at_val is None:
                    # Handle None value
                    needs_repair = True
                    repairs['created_at'] = datetime.utcnow().isoformat()
            
            # Check for corrupted keywords (should be list, not 'None' string)
            if 'keywords' in node_data:
                keywords_val = node_data['keywords']
                if isinstance(keywords_val, str):
                    if keywords_val.lower() == 'none' or keywords_val == '':
                        needs_repair = True
                        repairs['keywords'] = []
                    else:
                        # Try to parse as JSON
                        try:
                            import json
                            parsed = json.loads(keywords_val)
                            if not isinstance(parsed, list):
                                needs_repair = True
                                repairs['keywords'] = []
                        except:
                            # Invalid JSON - repair it
                            needs_repair = True
                            repairs['keywords'] = []
                elif keywords_val is None:
                    needs_repair = True
                    repairs['keywords'] = []
            
            # Check for corrupted tags (should be list, not 'None' string)
            if 'tags' in node_data:
                tags_val = node_data['tags']
                if isinstance(tags_val, str):
                    if tags_val.lower() == 'none' or tags_val == '':
                        needs_repair = True
                        repairs['tags'] = []
                    else:
                        # Try to parse as JSON
                        try:
                            import json
                            parsed = json.loads(tags_val)
                            if not isinstance(parsed, list):
                                needs_repair = True
                                repairs['tags'] = []
                        except:
                            # Invalid JSON - repair it
                            needs_repair = True
                            repairs['tags'] = []
                elif tags_val is None:
                    needs_repair = True
                    repairs['tags'] = []
            
            # Check for corrupted metadata (should be dict, not 'None' string)
            if 'metadata' in node_data:
                metadata_val = node_data['metadata']
                if isinstance(metadata_val, str):
                    if metadata_val.lower() == 'none' or metadata_val == '':
                        needs_repair = True
                        repairs['metadata'] = {}
                    else:
                        # Try to parse as JSON
                        try:
                            import json
                            parsed = json.loads(metadata_val)
                            if not isinstance(parsed, dict):
                                needs_repair = True
                                repairs['metadata'] = {}
                        except:
                            # Invalid JSON - repair it
                            needs_repair = True
                            repairs['metadata'] = {}
                elif metadata_val is None:
                    needs_repair = True
                    repairs['metadata'] = {}
            
            # Apply repairs if needed
            if needs_repair:
                try:
                    # Update node data
                    for key, value in repairs.items():
                        node_data[key] = value
                    
                    # Try to create AtomicNote to validate
                    try:
                        note = AtomicNote(**node_data)
                        # Update node in graph
                        graph.graph.nodes[node_id].update(repairs)
                        repaired_count += 1
                        log_event("NODE_REPAIRED", {
                            "node_id": node_id,
                            "repairs": repairs
                        })
                    except Exception as e:
                        log_debug(f"Failed to repair node {node_id}: {e}")
                        # If repair failed, try to fix created_at at least
                        if 'created_at' in repairs:
                            node_data['created_at'] = repairs['created_at']
                            graph.graph.nodes[node_id]['created_at'] = repairs['created_at']
                except Exception as e:
                    log_debug(f"Error repairing node {node_id}: {e}")
        except Exception as e:
            log_debug(f"Error checking node {node_id} for corruption: {e}")
            continue
    
    return repaired_count


def validate_note_types(
    notes: Dict[str, AtomicNote],
    graph: GraphStore,
    llm_service: Optional[LLMService] = None,
    ignore_flags: bool = False
) -> Dict[str, Any]:
    """
    Validiert und korrigiert Note-Types.
    
    Prüft ob Type vorhanden und gültig ist (rule, procedure, concept, tool, reference, integration).
    Korrigiert ungültige Types basierend auf Content.
    
    Args:
        notes: Dict von note_id -> AtomicNote
        graph: GraphStore Instanz
        llm_service: Optional LLMService für Type-Erkennung
        ignore_flags: Ignoriere Flags (default: False)
    
    Returns:
        Dict mit:
        - "types_validated": Anzahl validierte Types
        - "types_corrected": Anzahl korrigierte Types
        - "invalid_types": Liste von Node-IDs mit ungültigen Types
    """
    if len(notes) == 0:
        return {
            "types_validated": 0,
            "types_corrected": 0,
            "invalid_types": []
        }
    
    valid_types = ["rule", "procedure", "concept", "tool", "reference", "integration"]
    validated_count = 0
    corrected_count = 0
    invalid_types = []
    now_iso = datetime.utcnow().isoformat()
    
    for node_id, note in notes.items():
        if node_id not in graph.graph.nodes:
            continue
        
        # Initialisiere Metadata falls nicht vorhanden
        if not note.metadata:
            note.metadata = {}
            graph.graph.nodes[node_id]["metadata"] = {}
        
        metadata = note.metadata
        type_flag = metadata.get("type_validated_at")
        should_validate = not note.type or note.type not in valid_types or ignore_flags or _is_flag_too_old(type_flag, 30)
        
        if should_validate:
            if not note.type or note.type not in valid_types:
                invalid_types.append(node_id)
                
                # Auto-Korrektur: Bestimme Type basierend auf Content
                if llm_service and note.content and len(note.content.strip()) >= 50:
                    try:
                        prompt = f"""Bestimme den Type dieser Note basierend auf dem Content. Mögliche Types: rule, procedure, concept, tool, reference, integration.

Content:
{note.content[:500]}

Antworte nur mit einem der Types: rule, procedure, concept, tool, reference, integration"""
                        response = llm_service._call_llm(prompt)
                        response_lower = response.strip().lower()
                        
                        # Extrahiere Type aus Response
                        detected_type = None
                        for valid_type in valid_types:
                            if valid_type in response_lower:
                                detected_type = valid_type
                                break
                        
                        if not detected_type:
                            # Fallback: Bestimme Type basierend auf Keywords/Content
                            content_lower = note.content.lower()
                            if any(kw in content_lower for kw in ["how to", "anleitung", "schritt", "tutorial", "setup", "install"]):
                                detected_type = "procedure"
                            elif any(kw in content_lower for kw in ["tool", "werkzeug", "utility", "script", "command"]):
                                detected_type = "tool"
                            elif any(kw in content_lower for kw in ["regel", "rule", "guideline", "best practice", "should", "must"]):
                                detected_type = "rule"
                            elif any(kw in content_lower for kw in ["konzept", "concept", "theorie", "theory", "idea", "principle"]):
                                detected_type = "concept"
                            elif any(kw in content_lower for kw in ["referenz", "reference", "dokumentation", "docs", "api", "spec"]):
                                detected_type = "reference"
                            else:
                                detected_type = "concept"  # Default
                        
                        if detected_type:
                            old_type = note.type
                            note.type = detected_type
                            graph.graph.nodes[node_id]["type"] = detected_type
                            metadata["type_validated_at"] = now_iso
                            graph.graph.nodes[node_id]["metadata"] = metadata
                            corrected_count += 1
                            log_event("NOTE_TYPE_CORRECTED", {
                                "node_id": node_id,
                                "old_type": old_type,
                                "new_type": detected_type
                            })
                    except Exception as e:
                        log_debug(f"Error correcting type for {node_id}: {e}")
                        # Fallback: Setze Default-Type
                        if not note.type:
                            note.type = "concept"
                            graph.graph.nodes[node_id]["type"] = "concept"
                            metadata["type_validated_at"] = now_iso
                            graph.graph.nodes[node_id]["metadata"] = metadata
                            corrected_count += 1
                else:
                    # Kein LLM, setze Default-Type
                    if not note.type:
                        note.type = "concept"
                        graph.graph.nodes[node_id]["type"] = "concept"
                        metadata["type_validated_at"] = now_iso
                        graph.graph.nodes[node_id]["metadata"] = metadata
                        corrected_count += 1
            
            # Setze Flag auch wenn Type bereits gültig war
            metadata["type_validated_at"] = now_iso
            graph.graph.nodes[node_id]["metadata"] = metadata
            validated_count += 1
    
    if validated_count > 0 or corrected_count > 0:
        log_event("NOTE_TYPES_VALIDATED", {
            "validated": validated_count,
            "corrected": corrected_count,
            "invalid_count": len(invalid_types)
        })
    
    return {
        "types_validated": validated_count,
        "types_corrected": corrected_count,
        "invalid_types": invalid_types
    }


def temporal_note_cleanup(
    notes: Dict[str, AtomicNote],
    graph: GraphStore,
    max_age_days: int = 365,
    archive_instead_of_delete: bool = True
) -> Dict[str, Any]:
    """
    Prüft Notes älter als max_age_days und archiviert/löscht sie optional.
    
    Args:
        notes: Dict von note_id -> AtomicNote
        graph: GraphStore Instanz
        max_age_days: Maximale Alter in Tagen (default: 365 = 1 Jahr)
        archive_instead_of_delete: Wenn True, markiert Notes als archiviert statt zu löschen (default: True)
    
    Returns:
        Dict mit:
        - "notes_checked": Anzahl geprüfte Notes
        - "notes_archived": Anzahl archivierte Notes
        - "notes_deleted": Anzahl gelöschte Notes (wenn archive_instead_of_delete=False)
        - "old_notes": Liste von Node-IDs mit alten Notes
    """
    if len(notes) == 0:
        return {
            "notes_checked": 0,
            "notes_archived": 0,
            "notes_deleted": 0,
            "old_notes": []
        }
    
    now = datetime.utcnow()
    old_notes = []
    archived_count = 0
    deleted_count = 0
    
    for node_id, note in notes.items():
        if node_id not in graph.graph.nodes:
            continue
        
        # Prüfe created_at
        if note.created_at:
            try:
                if isinstance(note.created_at, str):
                    created_date = datetime.fromisoformat(note.created_at.replace('Z', '+00:00'))
                else:
                    created_date = note.created_at
                
                age_days = (now - created_date.replace(tzinfo=None)).days
                
                if age_days > max_age_days:
                    old_notes.append({
                        "id": node_id,
                        "age_days": age_days,
                        "created_at": note.created_at.isoformat() if hasattr(note.created_at, 'isoformat') else str(note.created_at),
                        "summary": note.contextual_summary[:100] if note.contextual_summary else "N/A"
                    })
                    
                    if archive_instead_of_delete:
                        # Markiere als archiviert
                        if not note.metadata:
                            note.metadata = {}
                            graph.graph.nodes[node_id]["metadata"] = {}
                        
                        note.metadata["archived"] = True
                        note.metadata["archived_at"] = now.isoformat()
                        graph.graph.nodes[node_id]["metadata"] = note.metadata
                        archived_count += 1
                        
                        log_event("NOTE_ARCHIVED", {
                            "node_id": node_id,
                            "age_days": age_days
                        })
                    else:
                        # Lösche Note (nur wenn keine wichtigen Edges vorhanden)
                        degree = graph.graph.degree(node_id)
                        if degree == 0:  # Nur isolierte Notes löschen
                            graph.graph.remove_node(node_id)
                            deleted_count += 1
                            log_event("NOTE_DELETED_TEMPORAL", {
                                "node_id": node_id,
                                "age_days": age_days
                            })
            except Exception as e:
                log_debug(f"Error checking age for {node_id}: {e}")
                continue
    
    if archived_count > 0 or deleted_count > 0:
        log_event("TEMPORAL_CLEANUP_COMPLETED", {
            "checked": len(notes),
            "archived": archived_count,
            "deleted": deleted_count,
            "old_notes_count": len(old_notes)
        })
    
    return {
        "notes_checked": len(notes),
        "notes_archived": archived_count,
        "notes_deleted": deleted_count,
        "old_notes": old_notes[:10]  # Max 10 für Response
    }


def calculate_graph_health_score(
    notes: Dict[str, AtomicNote],
    graph: GraphStore
) -> Dict[str, Any]:
    """
    Berechnet einen Gesamt-Health-Score für den Graph (0.0 - 1.0).
    
    Bewertet:
    - Durchschnittlicher Quality-Score aller Notes
    - Graph-Connectivity (Anteil isolierter Nodes)
    - Edge-Qualität (Anteil Edges mit Reasoning)
    - Note-Vollständigkeit (Anteil Notes mit allen Feldern)
    
    Args:
        notes: Dict von note_id -> AtomicNote
        graph: GraphStore Instanz
    
    Returns:
        Dict mit:
        - "health_score": Gesamt-Score (0.0 - 1.0)
        - "health_level": "excellent" | "good" | "fair" | "poor" | "very_poor"
        - "average_quality_score": Durchschnittlicher Quality-Score
        - "connectivity_score": Connectivity-Score (0.0 - 1.0)
        - "edge_quality_score": Edge-Quality-Score (0.0 - 1.0)
        - "completeness_score": Completeness-Score (0.0 - 1.0)
        - "isolated_nodes_ratio": Anteil isolierter Nodes (0.0 - 1.0)
        - "edges_with_reasoning_ratio": Anteil Edges mit Reasoning (0.0 - 1.0)
    """
    if len(notes) == 0:
        return {
            "health_score": 0.0,
            "health_level": "very_poor",
            "average_quality_score": 0.0,
            "connectivity_score": 0.0,
            "edge_quality_score": 0.0,
            "completeness_score": 0.0,
            "isolated_nodes_ratio": 1.0,
            "edges_with_reasoning_ratio": 0.0
        }
    
    # 1. Durchschnittlicher Quality-Score
    quality_scores = []
    for node_id, note in notes.items():
        if node_id not in graph.graph.nodes:
            continue
        
        metadata = note.metadata or {}
        quality_score = metadata.get("quality_score")
        if quality_score is not None:
            quality_scores.append(quality_score)
    
    average_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    
    # 2. Connectivity-Score (Anteil nicht-isolierter Nodes)
    total_nodes = len(notes)
    isolated_count = 0
    for node_id in notes.keys():
        if node_id in graph.graph.nodes:
            degree = graph.graph.degree(node_id)
            if degree == 0:
                isolated_count += 1
    
    isolated_ratio = isolated_count / total_nodes if total_nodes > 0 else 1.0
    connectivity_score = 1.0 - isolated_ratio
    
    # 3. Edge-Quality-Score (Anteil Edges mit Reasoning)
    total_edges = len(list(graph.graph.edges()))
    edges_with_reasoning = 0
    
    for source, target, data in graph.graph.edges(data=True):
        if data:
            reasoning = data.get("reasoning") or data.get("reasoning_text")
            if reasoning and reasoning.strip() and reasoning.lower() != "no reasoning provided":
                edges_with_reasoning += 1
    
    edges_with_reasoning_ratio = edges_with_reasoning / total_edges if total_edges > 0 else 0.0
    edge_quality_score = edges_with_reasoning_ratio
    
    # 4. Completeness-Score (Anteil Notes mit allen Feldern)
    complete_notes = 0
    for node_id, note in notes.items():
        has_content = note.content and len(note.content.strip()) > 0
        has_summary = note.contextual_summary and len(note.contextual_summary.strip()) > 0
        has_keywords = note.keywords and len(note.keywords) >= 2
        has_tags = note.tags and len(note.tags) >= 1
        has_type = note.type and note.type in ["rule", "procedure", "concept", "tool", "reference", "integration"]
        
        if has_content and has_summary and has_keywords and has_tags and has_type:
            complete_notes += 1
    
    completeness_score = complete_notes / total_nodes if total_nodes > 0 else 0.0
    
    # Gesamt-Health-Score (gewichteter Durchschnitt)
    health_score = (
        average_quality * 0.40 +  # 40% Quality
        connectivity_score * 0.25 +  # 25% Connectivity
        edge_quality_score * 0.20 +  # 20% Edge Quality
        completeness_score * 0.15  # 15% Completeness
    )
    
    # Health-Level bestimmen
    if health_score >= 0.9:
        health_level = "excellent"
    elif health_score >= 0.75:
        health_level = "good"
    elif health_score >= 0.6:
        health_level = "fair"
    elif health_score >= 0.4:
        health_level = "poor"
    else:
        health_level = "very_poor"
    
    return {
        "health_score": round(health_score, 3),
        "health_level": health_level,
        "average_quality_score": round(average_quality, 3),
        "connectivity_score": round(connectivity_score, 3),
        "edge_quality_score": round(edge_quality_score, 3),
        "completeness_score": round(completeness_score, 3),
        "isolated_nodes_ratio": round(isolated_ratio, 3),
        "edges_with_reasoning_ratio": round(edges_with_reasoning_ratio, 3)
    }


def find_dead_end_nodes(
    notes: Dict[str, AtomicNote],
    graph: GraphStore
) -> Dict[str, Any]:
    """
    Findet Dead-End Nodes (Nodes nur mit eingehenden, aber keine ausgehenden Edges).
    
    Prüft die tatsächlichen Relations im Graph, nicht nur Graph-Metriken.
    
    Args:
        notes: Dict von note_id -> AtomicNote
        graph: GraphStore Instanz
    
    Returns:
        Dict mit:
        - "dead_end_nodes": Liste von Node-IDs mit Dead-Ends
        - "dead_end_count": Anzahl Dead-End Nodes
        - "dead_end_details": Liste mit Details (id, in_degree, out_degree, summary)
    """
    if len(notes) == 0:
        return {
            "dead_end_nodes": [],
            "dead_end_count": 0,
            "dead_end_details": []
        }
    
    dead_end_nodes = []
    dead_end_details = []
    
    # Sammle alle Edges für manuelle Prüfung
    incoming_edges = {}  # target -> [sources]
    outgoing_edges = {}  # source -> [targets]
    
    for source, target, data in graph.graph.edges(data=True):
        # Incoming edges
        if target not in incoming_edges:
            incoming_edges[target] = []
        incoming_edges[target].append(source)
        
        # Outgoing edges
        if source not in outgoing_edges:
            outgoing_edges[source] = []
        outgoing_edges[source].append(target)
    
    for node_id in notes.keys():
        if node_id not in graph.graph.nodes:
            continue
        
        # Zähle tatsächliche Relations
        incoming_count = len(incoming_edges.get(node_id, []))
        outgoing_count = len(outgoing_edges.get(node_id, []))
        total_degree = incoming_count + outgoing_count
        
        # Skip isolierte Nodes (werden bereits von find_isolated_nodes behandelt)
        if total_degree == 0:
            continue
        
        # Dead-End Detection:
        # Ein Dead-End ist ein Node, der:
        # - incoming_count > 0 (andere Nodes haben Edges ZU diesem Node)
        # - outgoing_count == 0 (dieser Node hat KEINE ausgehenden Edges)
        # Das bedeutet: Wissen fließt ZU diesem Node, aber nicht weiter
        is_dead_end = incoming_count > 0 and outgoing_count == 0
        
        if is_dead_end:
            dead_end_nodes.append(node_id)
            note = notes.get(node_id)
            dead_end_details.append({
                "id": node_id,
                "in_degree": incoming_count,
                "out_degree": outgoing_count,
                "total_degree": total_degree,
                "summary": note.contextual_summary[:100] if note and note.contextual_summary else "N/A"
            })
    
    if dead_end_nodes:
        log_event("DEAD_END_NODES_FOUND", {
            "count": len(dead_end_nodes),
            "node_ids": dead_end_nodes[:10]  # Nur erste 10 loggen
        })
    
    return {
        "dead_end_nodes": dead_end_nodes,
        "dead_end_count": len(dead_end_nodes),
        "dead_end_details": dead_end_details[:10]  # Max 10 für Response
    }


def _validate_enzyme_parameters(
    prune_config: Optional[Dict[str, Any]] = None,
    suggest_config: Optional[Dict[str, Any]] = None,
    refine_config: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], List[str]]:
    """
    Validiert und normalisiert Enzyme-Parameter.
    
    Args:
        prune_config: Config für prune_links
        suggest_config: Config für suggest_relations
        refine_config: Config für refine_summaries
    
    Returns:
        Tuple von (validated_prune_config, validated_suggest_config, validated_refine_config, validation_warnings)
        validation_warnings: Liste von Warnungen über korrigierte Parameter
    """
    warnings = []
    # Default values
    DEFAULT_PRUNE_MAX_AGE_DAYS = 90
    DEFAULT_PRUNE_MIN_WEIGHT = 0.3
    DEFAULT_SUGGEST_THRESHOLD = 0.75
    DEFAULT_SUGGEST_MAX = 10
    DEFAULT_REFINE_SIMILARITY_THRESHOLD = 0.75
    DEFAULT_REFINE_MAX = 10
    
    # Validate prune_config (support both naming conventions)
    validated_prune = {}
    if prune_config:
        # prune_max_age_days / max_age_days: >= 0 (default: 90)
        max_age = prune_config.get("prune_max_age_days") or prune_config.get("max_age_days", DEFAULT_PRUNE_MAX_AGE_DAYS)
        if not isinstance(max_age, (int, float)) or max_age < 0:
            warning = f"Invalid prune_max_age_days: {max_age} (must be >= 0), using default: {DEFAULT_PRUNE_MAX_AGE_DAYS}"
            log_debug(warning)
            warnings.append(warning)
            validated_prune["prune_max_age_days"] = DEFAULT_PRUNE_MAX_AGE_DAYS
        else:
            validated_prune["prune_max_age_days"] = int(max_age)
        
        # prune_min_weight / min_weight: 0.0-1.0 (default: 0.3)
        min_weight = prune_config.get("prune_min_weight") or prune_config.get("min_weight", DEFAULT_PRUNE_MIN_WEIGHT)
        if not isinstance(min_weight, (int, float)) or min_weight < 0.0 or min_weight > 1.0:
            warning = f"Invalid prune_min_weight: {min_weight} (must be 0.0-1.0), using default: {DEFAULT_PRUNE_MIN_WEIGHT}"
            log_debug(warning)
            warnings.append(warning)
            validated_prune["prune_min_weight"] = DEFAULT_PRUNE_MIN_WEIGHT
        else:
            validated_prune["prune_min_weight"] = float(min_weight)
    else:
        validated_prune = {
            "prune_max_age_days": DEFAULT_PRUNE_MAX_AGE_DAYS,
            "prune_min_weight": DEFAULT_PRUNE_MIN_WEIGHT
        }
    
    # Validate suggest_config (support both naming conventions)
    validated_suggest = {}
    if suggest_config:
        # suggest_threshold / threshold: 0.0-1.0 (default: 0.75)
        threshold = suggest_config.get("suggest_threshold") or suggest_config.get("threshold", DEFAULT_SUGGEST_THRESHOLD)
        if not isinstance(threshold, (int, float)) or threshold < 0.0 or threshold > 1.0:
            warning = f"Invalid suggest_threshold: {threshold} (must be 0.0-1.0), using default: {DEFAULT_SUGGEST_THRESHOLD}"
            log_debug(warning)
            warnings.append(warning)
            validated_suggest["suggest_threshold"] = DEFAULT_SUGGEST_THRESHOLD
        else:
            validated_suggest["suggest_threshold"] = float(threshold)
        
        # suggest_max / max_suggestions: >= 0, max 1000 (default: 10)
        suggest_max = suggest_config.get("suggest_max") or suggest_config.get("max_suggestions", DEFAULT_SUGGEST_MAX)
        if not isinstance(suggest_max, (int, float)) or suggest_max < 0:
            warning = f"Invalid suggest_max: {suggest_max} (must be >= 0), using default: {DEFAULT_SUGGEST_MAX}"
            log_debug(warning)
            warnings.append(warning)
            validated_suggest["suggest_max"] = DEFAULT_SUGGEST_MAX
        elif suggest_max > 1000:
            warning = f"suggest_max too high: {suggest_max} (max: 1000), capping at 1000"
            log_debug(warning)
            warnings.append(warning)
            validated_suggest["suggest_max"] = 1000
        else:
            validated_suggest["suggest_max"] = int(suggest_max)
    else:
        validated_suggest = {
            "suggest_threshold": DEFAULT_SUGGEST_THRESHOLD,
            "suggest_max": DEFAULT_SUGGEST_MAX
        }
    
    # Validate refine_config (support both naming conventions)
    validated_refine = {}
    if refine_config:
        # refine_similarity_threshold / similarity_threshold: 0.0-1.0 (default: 0.75)
        similarity_threshold = refine_config.get("refine_similarity_threshold") or refine_config.get("similarity_threshold", DEFAULT_REFINE_SIMILARITY_THRESHOLD)
        if not isinstance(similarity_threshold, (int, float)) or similarity_threshold < 0.0 or similarity_threshold > 1.0:
            warning = f"Invalid refine_similarity_threshold: {similarity_threshold} (must be 0.0-1.0), using default: {DEFAULT_REFINE_SIMILARITY_THRESHOLD}"
            log_debug(warning)
            warnings.append(warning)
            validated_refine["refine_similarity_threshold"] = DEFAULT_REFINE_SIMILARITY_THRESHOLD
        else:
            validated_refine["refine_similarity_threshold"] = float(similarity_threshold)
        
        # refine_max / max_refinements: >= 0, max 1000 (default: 10)
        refine_max = refine_config.get("refine_max") or refine_config.get("max_refinements", DEFAULT_REFINE_MAX)
        if not isinstance(refine_max, (int, float)) or refine_max < 0:
            warning = f"Invalid refine_max: {refine_max} (must be >= 0), using default: {DEFAULT_REFINE_MAX}"
            log_debug(warning)
            warnings.append(warning)
            validated_refine["refine_max"] = DEFAULT_REFINE_MAX
        elif refine_max > 1000:
            warning = f"refine_max too high: {refine_max} (max: 1000), capping at 1000"
            log_debug(warning)
            warnings.append(warning)
            validated_refine["refine_max"] = 1000
        else:
            validated_refine["refine_max"] = int(refine_max)
    else:
        validated_refine = {
            "refine_similarity_threshold": DEFAULT_REFINE_SIMILARITY_THRESHOLD,
            "refine_max": DEFAULT_REFINE_MAX
        }
    
    return validated_prune, validated_suggest, validated_refine, warnings


def run_memory_enzymes(
    graph: GraphStore,
    llm_service: LLMService,
    prune_config: Optional[Dict[str, Any]] = None,
    suggest_config: Optional[Dict[str, Any]] = None,
    refine_config: Optional[Dict[str, Any]] = None,
    auto_add_suggestions: bool = False,
    ignore_flags: bool = False
) -> Dict[str, Any]:
    """
    Führt alle Memory-Enzyme aus.
    
    Args:
        graph: GraphStore Instanz
        llm_service: LLMService Instanz
        prune_config: Config für prune_links (optional)
            - prune_max_age_days: >= 0 (default: 90) - Entfernt Edges älter als X Tage
            - prune_min_weight: 0.0-1.0 (default: 0.3) - Behält nur Edges mit Gewicht >= X
        suggest_config: Config für suggest_relations (optional)
            - suggest_threshold: 0.0-1.0 (default: 0.75) - Mindest-Ähnlichkeit für Relation-Vorschläge
            - suggest_max: 0-1000 (default: 10) - Maximale Anzahl Vorschläge
        refine_config: Config für refine_summaries (optional)
            - refine_similarity_threshold: 0.0-1.0 (default: 0.75) - Mindest-Ähnlichkeit für Summary-Refinement
            - refine_max: 0-1000 (default: 10) - Maximale Anzahl zu refinender Summaries
        auto_add_suggestions: Wenn True, werden vorgeschlagene Relations automatisch hinzugefügt (default: False)
        ignore_flags: Wenn True, ignoriert Validierungs-Flags (default: False)
    
    Returns:
        Dict mit Ergebnissen (inkl. "relations_auto_added" wenn auto_add_suggestions=True)
    """
    results = {
        "pruned_count": 0,
        "zombie_nodes_removed": 0,
        "low_quality_notes_removed": 0,  # Anzahl entfernte irrelevante Notes
        "self_loops_removed": 0,  # Anzahl entfernte Self-Loops
        "isolated_nodes_found": 0,  # Anzahl isolierter Nodes
        "isolated_nodes": [],  # Liste der isolierten Nodes mit Details
        "isolated_nodes_linked": 0,  # Anzahl automatisch verlinkter isolierter Nodes
        "duplicates_merged": 0,  # Anzahl gemergte Duplikate
        "suggestions_count": 0,
        "suggestions": [],  # Liste der vorgeschlagenen Relations
        "relations_auto_added": 0,  # Anzahl automatisch hinzugefügter Relations
        "digested_count": 0,
        "summaries_refined": 0,
        "notes_validated": 0,  # Anzahl validierte Notes
        "notes_corrected": 0,  # Anzahl korrigierte Notes
        "corrupted_nodes_repaired": 0,  # Anzahl reparierte korrupte Nodes
        "quality_scores_calculated": 0,  # Anzahl berechneter Quality-Scores
        "low_quality_notes": [],  # Liste von Notes mit niedrigem Score (< 0.6)
        "keywords_normalized": 0,  # Anzahl normalisierte Keywords
        "keywords_removed": 0,  # Anzahl entfernte Keywords
        "keywords_corrected": 0,  # Anzahl korrigierte Keywords
        "edges_validated": 0,  # Anzahl validierte Edges
        "edges_removed": 0,  # Anzahl entfernte Edges (schwach/ohne Reasoning)
        "reasonings_added": 0,  # Anzahl ergänzte Reasoning-Felder
        "types_standardized": 0,  # Anzahl standardisierte Relation Types
        "validation_warnings": [],  # Liste von Warnungen über korrigierte Parameter
        # Neue Funktionen
        "types_validated": 0,  # Anzahl validierte Note-Types
        "types_corrected": 0,  # Anzahl korrigierte Note-Types
        "notes_archived": 0,  # Anzahl archivierte Notes (temporal cleanup)
        "notes_deleted": 0,  # Anzahl gelöschte Notes (temporal cleanup)
        "graph_health_score": 0.0,  # Graph Health Score (0.0 - 1.0)
        "graph_health_level": "unknown",  # Graph Health Level
        "dead_end_nodes_found": 0,  # Anzahl Dead-End Nodes
        "dead_end_nodes": []  # Liste von Dead-End Nodes mit Details
    }
    
    # Validate and normalize parameters FIRST
    validated_prune, validated_suggest, validated_refine, validation_warnings = _validate_enzyme_parameters(
        prune_config, suggest_config, refine_config
    )
    
    # Add validation warnings to results
    results["validation_warnings"] = validation_warnings
    
    # 0. Repair Corrupted Nodes (FIRST - before any other operations)
    results["corrupted_nodes_repaired"] = repair_corrupted_nodes(graph)
    
    # 1. Prune Links (use validated parameters)
    results["pruned_count"] = prune_links(
        graph,
        max_age_days=validated_prune.get("prune_max_age_days", 90),
        min_weight=validated_prune.get("prune_min_weight", 0.3)
    )
    
    # 1.5. Prune Zombie Nodes (nach Links, damit keine orphaned Edges entstehen)
    results["zombie_nodes_removed"] = prune_zombie_nodes(graph)
    
    # 1.5.5. Remove Low Quality Notes (CAPTCHA, Fehlerseiten, etc.)
    # Sammle Notes für Quality-Check
    notes_for_quality = {}
    for node_id in graph.graph.nodes():
        node_data = graph.graph.nodes[node_id]
        try:
            note = AtomicNote(**node_data)
            notes_for_quality[node_id] = note
        except Exception:
            continue
    
    if notes_for_quality:
        results["low_quality_notes_removed"] = remove_low_quality_notes(notes_for_quality, graph, llm_service)
    
    # 1.6. Remove Self-Loops (vor Duplikat-Merge)
    results["self_loops_removed"] = remove_self_loops(graph)
    
    # 1.6.5. Validate and Fix Edges (Reasoning, Types, schwache Edges)
    # Sammle Notes für Edge-Validierung
    notes_for_edge_validation = {}
    for node_id in graph.graph.nodes():
        node_data = graph.graph.nodes[node_id]
        try:
            note = AtomicNote(**node_data)
            notes_for_edge_validation[node_id] = note
        except Exception:
            continue
    
    if notes_for_edge_validation:
        edge_validation_results = validate_and_fix_edges(
            graph,
            notes_for_edge_validation,
            llm_service,
            min_weight_for_reasoning=0.65,
            ignore_flags=ignore_flags
        )
        results["edges_removed"] = edge_validation_results["edges_removed"]
        results["reasonings_added"] = edge_validation_results["reasonings_added"]
        results["types_standardized"] = edge_validation_results["types_standardized"]
        results["edges_validated"] = len(list(graph.graph.edges()))
    
    # 1.6. Merge Duplicates (vor Relations, damit keine doppelten Suggestions entstehen)
    # Sammle alle Notes für Duplikat-Check
    notes_for_merge = {}
    for node_id in graph.graph.nodes():
        node_data = graph.graph.nodes[node_id]
        try:
            note = AtomicNote(**node_data)
            notes_for_merge[node_id] = note
        except Exception:
            continue  # Skip invalid nodes
    
    if len(notes_for_merge) >= 2:
        results["duplicates_merged"] = merge_duplicates(
            notes_for_merge, 
            graph, 
            llm_service=llm_service,
            content_similarity_threshold=0.98
        )
    
    # 1.7. Normalize and Clean Keywords
    # Sammle Notes für Keyword-Normalisierung
    notes_for_keywords = {}
    for node_id in graph.graph.nodes():
        node_data = graph.graph.nodes[node_id]
        try:
            note = AtomicNote(**node_data)
            notes_for_keywords[node_id] = note
        except Exception:
            continue
    
    if notes_for_keywords:
        keywords_results = normalize_and_clean_keywords(
            notes_for_keywords, 
            graph, 
            llm_service, 
            max_keywords=7,
            ignore_flags=ignore_flags
        )
        results["keywords_normalized"] = keywords_results["keywords_normalized"]
        results["keywords_removed"] = keywords_results["keywords_removed"]
        results["keywords_corrected"] = keywords_results["keywords_corrected"]
    
    # 1.7.5. Validate Note Types (vor validate_notes, damit Types korrekt sind)
    # Sammle Notes für Type-Validierung
    notes_for_type_validation = {}
    for node_id in graph.graph.nodes():
        node_data = graph.graph.nodes[node_id]
        try:
            note = AtomicNote(**node_data)
            notes_for_type_validation[node_id] = note
        except Exception:
            continue
    
    if notes_for_type_validation:
        type_validation_results = validate_note_types(
            notes_for_type_validation,
            graph,
            llm_service,
            ignore_flags=ignore_flags
        )
        results["types_validated"] = type_validation_results["types_validated"]
        results["types_corrected"] = type_validation_results["types_corrected"]
    
    # 1.8. Validate Notes (prüft Vollständigkeit und korrigiert)
    # Sammle alle Notes für Validierung
    notes_for_validation = {}
    for node_id in graph.graph.nodes():
        node_data = graph.graph.nodes[node_id]
        try:
            note = AtomicNote(**node_data)
            notes_for_validation[node_id] = note
        except Exception:
            continue
    
    if notes_for_validation:
        validation_results = validate_notes(
            notes_for_validation, 
            graph, 
            llm_service,
            ignore_flags=ignore_flags
        )
        results["notes_validated"] = validation_results["validated"]
        results["notes_corrected"] = validation_results["corrected"]
        
        # Sammle Notes mit niedrigem Quality-Score (nach validate_notes, da Scores dort berechnet werden)
        low_quality_notes = []
        for node_id in graph.graph.nodes():
            node_data = graph.graph.nodes[node_id]
            # Handle None node_data gracefully
            if node_data is None:
                node_data = {}
            metadata = node_data.get("metadata", {})
            # Handle None metadata gracefully
            if metadata is None:
                metadata = {}
            quality_score = metadata.get("quality_score")
            if quality_score is not None:
                if quality_score < 0.6:
                    try:
                        note = AtomicNote(**node_data)
                        low_quality_notes.append({
                            "id": node_id,
                            "score": quality_score,
                            "level": metadata.get("quality_level", "unknown"),
                            "summary": note.contextual_summary[:100] if note.contextual_summary else "N/A",
                            "issues": metadata.get("quality_issues", [])
                        })
                    except Exception:
                        continue
        
        # Zähle wie viele Quality-Scores berechnet wurden
        quality_scores_count = 0
        for node_id in graph.graph.nodes():
            node_data = graph.graph.nodes[node_id]
            # Handle None node_data gracefully
            if node_data is None:
                node_data = {}
            metadata = node_data.get("metadata", {})
            # Handle None metadata gracefully
            if metadata is None:
                metadata = {}
            if "quality_score" in metadata:
                quality_scores_count += 1
        
        results["quality_scores_calculated"] = quality_scores_count
        results["low_quality_notes"] = sorted(low_quality_notes, key=lambda x: x["score"])[:10]  # Top 10 schlechteste
    
    # 1.8. Find Isolated Nodes (nach Validierung, für Auto-Linking)
    # Sammle Notes erneut für Isolated-Check (nach Validierung)
    notes_for_isolated = {}
    for node_id in graph.graph.nodes():
        node_data = graph.graph.nodes[node_id]
        try:
            note = AtomicNote(**node_data)
            notes_for_isolated[node_id] = note
        except Exception:
            continue
    
    isolated_nodes = find_isolated_nodes(notes_for_isolated, graph)
    results["isolated_nodes_found"] = len(isolated_nodes)
    # Speichere isolierte Nodes mit Details für Rückgabe
    results["isolated_nodes"] = [
        {
            "id": node_id,
            "summary": notes_for_isolated[node_id].contextual_summary if node_id in notes_for_isolated else "N/A",
            "type": notes_for_isolated[node_id].type if node_id in notes_for_isolated else "N/A",
            "tags": notes_for_isolated[node_id].tags if node_id in notes_for_isolated else []
        }
        for node_id in isolated_nodes[:10]  # Max 10 für Response
    ]
    
    # 1.8.1. Auto-Link Isolated Nodes
    if isolated_nodes and llm_service:
        results["isolated_nodes_linked"] = link_isolated_nodes(
            isolated_nodes,
            notes_for_isolated,
            graph,
            llm_service,
            similarity_threshold=0.70,
            max_links_per_node=3
        )
    
    if isolated_nodes:
        log_event("ISOLATED_NODES_FOUND", {
            "count": len(isolated_nodes),
            "node_ids": isolated_nodes[:10]  # Nur erste 10 loggen
        })
    
    # 2. Suggest Relations
    # Sammle alle Notes (erneut, da Duplikate entfernt wurden)
    notes = {}
    for node_id in graph.graph.nodes():
        node_data = graph.graph.nodes[node_id]
        try:
            # Versuche AtomicNote zu erstellen
            note = AtomicNote(**node_data)
            notes[node_id] = note
        except Exception:
            continue  # Skip invalid nodes
    
    if len(notes) >= 2:
        # 2.1. Refine Summaries (use validated parameters)
        results["summaries_refined"] = refine_summaries(
            notes,
            graph,
            llm_service,
            similarity_threshold=validated_refine.get("refine_similarity_threshold", 0.75),
            max_refinements=validated_refine.get("refine_max", 10)
        )
        
        # 2.2. Suggest Relations (use validated parameters)
        suggestions = suggest_relations(
            notes,
            graph,
            llm_service,
            threshold=validated_suggest.get("suggest_threshold", 0.75),
            max_suggestions=validated_suggest.get("suggest_max", 10)
        )
        results["suggestions_count"] = len(suggestions)
        # Speichere Suggestions im Ergebnis für Rückgabe
        results["suggestions"] = [
            {
                "from_id": s[0],
                "to_id": s[1],
                "similarity": float(s[2]),
                "from_summary": notes[s[0]].contextual_summary if s[0] in notes else "N/A",
                "to_summary": notes[s[1]].contextual_summary if s[1] in notes else "N/A"
            }
            for s in suggestions
        ]
        
        # Auto-Add Suggestions (wenn Parameter gesetzt)
        relations_added = 0
        if auto_add_suggestions and suggestions:
            for from_id, to_id, similarity in suggestions:
                try:
                    note_a = notes[from_id]
                    note_b = notes[to_id]
                    
                    # Erstelle Relation basierend auf Similarity
                    # Bestimme Relation-Type basierend auf Similarity und Content
                    if similarity >= 0.95:
                        relation_type = "relates_to"  # Sehr ähnlich = direkt verwandt
                    elif similarity >= 0.85:
                        relation_type = "supports"  # Ähnlich = unterstützt
                    else:
                        relation_type = "relates_to"  # Standard
                    
                    # Erstelle NoteRelation
                    relation = NoteRelation(
                        source_id=from_id,
                        target_id=to_id,
                        relation_type=relation_type,
                        reasoning=f"Auto-linked by enzymes based on high similarity ({similarity:.3f})",
                        weight=float(similarity)
                    )
                    
                    # Füge Edge zum Graph hinzu
                    graph.add_edge(relation)
                    relations_added += 1
                    
                    log_event("RELATION_AUTO_ADDED", {
                        "from": from_id,
                        "to": to_id,
                        "similarity": similarity,
                        "relation_type": relation_type
                    })
                except Exception as e:
                    log_debug(f"Error auto-adding relation {from_id} -> {to_id}: {e}")
                    continue
        
        # Logge Suggestions (aber füge sie nicht automatisch hinzu - User entscheidet)
        if suggestions:
            log_event("RELATIONS_SUGGESTED", {
                "count": len(suggestions),
                "auto_added": relations_added if auto_add_suggestions else 0,
                "suggestions": [
                    {"from": s[0], "to": s[1], "similarity": s[2]}
                    for s in suggestions[:5]  # Nur erste 5 loggen
                ]
            })
        
        results["relations_auto_added"] = relations_added
    
    # 3. Digest Nodes (optional, für später)
    # Finde Nodes mit vielen Children
    for node_id in graph.graph.nodes():
        neighbors = graph.get_neighbors(node_id)
        if len(neighbors) > 8:  # max_children default
            # Konvertiere zu AtomicNote Liste
            child_notes = []
            for neighbor_data in neighbors:
                try:
                    child_note = AtomicNote(**neighbor_data)
                    child_notes.append(child_note)
                except Exception:
                    continue
            
            if child_notes:
                summary = digest_node(node_id, child_notes, llm_service)
                if summary:
                    results["digested_count"] += 1
    
    # 4. Temporal Note Cleanup (prüft alte Notes)
    # Sammle alle Notes erneut für Temporal-Cleanup
    notes_for_temporal = {}
    for node_id in graph.graph.nodes():
        node_data = graph.graph.nodes[node_id]
        try:
            note = AtomicNote(**node_data)
            notes_for_temporal[node_id] = note
        except Exception:
            continue
    
    if notes_for_temporal:
        temporal_results = temporal_note_cleanup(
            notes_for_temporal,
            graph,
            max_age_days=365,  # 1 Jahr
            archive_instead_of_delete=True
        )
        results["notes_archived"] = temporal_results["notes_archived"]
        results["notes_deleted"] = temporal_results["notes_deleted"]
    
    # 5. Calculate Graph Health Score
    # Sammle alle Notes erneut für Health-Score
    notes_for_health = {}
    for node_id in graph.graph.nodes():
        node_data = graph.graph.nodes[node_id]
        try:
            note = AtomicNote(**node_data)
            notes_for_health[node_id] = note
        except Exception:
            continue
    
    if notes_for_health:
        health_results = calculate_graph_health_score(notes_for_health, graph)
        results["graph_health_score"] = health_results["health_score"]
        results["graph_health_level"] = health_results["health_level"]
        # Füge detaillierte Health-Metriken hinzu
        results["graph_health_details"] = {
            "average_quality_score": health_results["average_quality_score"],
            "connectivity_score": health_results["connectivity_score"],
            "edge_quality_score": health_results["edge_quality_score"],
            "completeness_score": health_results["completeness_score"],
            "isolated_nodes_ratio": health_results["isolated_nodes_ratio"],
            "edges_with_reasoning_ratio": health_results["edges_with_reasoning_ratio"]
        }
    
    # 6. Find Dead-End Nodes
    # Sammle alle Notes erneut für Dead-End-Detection
    notes_for_dead_end = {}
    for node_id in graph.graph.nodes():
        node_data = graph.graph.nodes[node_id]
        try:
            note = AtomicNote(**node_data)
            notes_for_dead_end[node_id] = note
        except Exception:
            continue
    
    if notes_for_dead_end:
        dead_end_results = find_dead_end_nodes(notes_for_dead_end, graph)
        results["dead_end_nodes_found"] = dead_end_results["dead_end_count"]
        results["dead_end_nodes"] = dead_end_results["dead_end_details"]
    
    return results

