# RustworkX Setup & Usage

## 🚀 Installation

```bash
pip install rustworkx
```

Oder in `requirements.txt`:
```txt
rustworkx>=0.14.0
```

## ⚙️ Konfiguration

Setze in deiner `.env` Datei:
```env
GRAPH_BACKEND=rustworkx
```

## 📊 Performance

RustworkX ist **3x-100x schneller** als NetworkX:
- Graph Creation: 3x-10x schneller
- Graph Traversal: 10x-100x schneller
- Algorithmen (BFS, SSSP, PageRank): 40x-250x schneller

## 💾 Persistenz

RustworkX verwendet **GraphML** für Persistenz:
- Datei: `data/graph.graphml`
- Standard-Format für Graph-Daten
- Automatisches Load/Save bei GraphStore-Operationen

## 🔧 Verwendung

Die Implementierung ist vollständig kompatibel mit dem bestehenden GraphStore-Interface:

```python
from a_mem.storage.engine import create_graph_store

# Erstellt RustworkX GraphStore (wenn GRAPH_BACKEND=rustworkx)
graph = create_graph_store()

# Alle Methoden funktionieren wie gewohnt:
graph.add_node(note)
graph.add_edge(relation)
graph.get_neighbors(node_id)
graph.save_snapshot()
```

## 🛡️ Safe Graph Wrapper

RustworkX wird automatisch mit dem **Safe Graph Wrapper** umhüllt, der Edge Cases abfängt und Daten validiert:

**Features:**
- ✅ **Automatische Validierung & Sanitization**: Alle Daten werden vor dem Speichern validiert
- ✅ **Edge Case Handling**: Behandelt leere Felder, Unicode, None-Werte, korrupte Daten
- ✅ **Konsistente Deserialisierung**: JSON-Strings werden automatisch zu Lists/Dicts konvertiert
- ✅ **Fehlerbehandlung & Recovery**: Versucht automatisch mit minimalen Daten, wenn vollständige Daten fehlschlagen
- ✅ **Automatische Aktivierung**: Wird automatisch aktiviert, wenn RustworkX verwendet wird

**Was wird automatisch behandelt:**
- `created_at = 'None'` String → Wird zu aktuellem Datum
- `keywords = 'None'` String → Wird zu leere Liste
- `tags = 'None'` String → Wird zu leere Liste
- JSON-Strings in Feldern → Werden automatisch deserialisiert
- Fehlende Nodes bei Edge-Operationen → Automatische Validierung
- Korrupte Daten → Automatische Reparatur oder Fallback auf minimale Daten

**Code:** `src/a_mem/storage/safe_graph_wrapper.py`

## ✅ Windows-Kompatibilität

✅ **Vollständig Windows-kompatibel** - RustworkX läuft nativ auf Windows!

## 🔄 Migration von NetworkX

1. Installiere RustworkX: `pip install rustworkx`
2. Setze `GRAPH_BACKEND=rustworkx` in `.env`
3. Starte die Anwendung - GraphML wird automatisch erstellt
4. Bestehende NetworkX JSON-Daten müssen manuell migriert werden (falls nötig)

## 📚 Weitere Informationen

- [RustworkX GitHub](https://github.com/Qiskit/rustworkx)
- [GraphML Format](https://en.wikipedia.org/wiki/GraphML)
- RustworkX ist vollständig implementiert und funktioniert (siehe `src/a_mem/storage/rustworkx_store.py`)

