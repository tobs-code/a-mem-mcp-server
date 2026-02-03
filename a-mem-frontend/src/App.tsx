import { useState, useEffect } from 'react';
import { Note, MemoryStats, createNote, retrieveMemories, getMemoryStats, listNotes } from './services/api';
import './App.css';

// TypeScript interfaces for our data structures
interface Relation {
  source_id: string;
  target_id: string;
  relation_type: string;
  reasoning: string;
  weight: number;
}

const App = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [notes, setNotes] = useState<Note[]>([]);
  const [memories, setMemories] = useState<Note[]>([]);
  const [stats, setStats] = useState<MemoryStats | null>(null);
  const [newNote, setNewNote] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // Load initial data
  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      setIsLoading(true);
      const statsData = await getMemoryStats();
      setStats(statsData);
      
      const notesData = await listNotes();
      setNotes(notesData.notes || []);
    } catch (error) {
      console.error('Error loading initial data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCreateNote = async () => {
    if (!newNote.trim()) return;
    
    setIsLoading(true);
    try {
      const result = await createNote(newNote);
      console.log('Note created:', result);
      
      // Reload notes to show the new one
      const notesData = await listNotes();
      setNotes(notesData.notes || []);
      setNewNote('');
    } catch (error) {
      console.error('Error creating note:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    
    setIsLoading(true);
    try {
      const result = await retrieveMemories(searchQuery);
      console.log('Search results:', result);
      setMemories(result.results || []);
    } catch (error) {
      console.error('Error searching memories:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <div className="logo-section">
          <h1>🧠 A-MEM MCP Server</h1>
          <p>Agentic Memory System Control Panel</p>
        </div>
        
        <nav className="nav-tabs">
          <button 
            className={activeTab === 'dashboard' ? 'active' : ''} 
            onClick={() => setActiveTab('dashboard')}
          >
            Dashboard
          </button>
          <button 
            className={activeTab === 'notes' ? 'active' : ''} 
            onClick={() => setActiveTab('notes')}
          >
            Memory Notes
          </button>
          <button 
            className={activeTab === 'search' ? 'active' : ''} 
            onClick={() => setActiveTab('search')}
          >
            Search Memories
          </button>
          <button 
            className={activeTab === 'relations' ? 'active' : ''} 
            onClick={() => setActiveTab('relations')}
          >
            Relations
          </button>
          <button 
            className={activeTab === 'stats' ? 'active' : ''} 
            onClick={() => setActiveTab('stats')}
          >
            System Stats
          </button>
        </nav>
      </header>

      <main className="main-content">
        {activeTab === 'dashboard' && (
          <DashboardView stats={stats} isLoading={isLoading} />
        )}

        {activeTab === 'notes' && (
          <NotesView 
            newNote={newNote} 
            setNewNote={setNewNote} 
            handleCreateNote={handleCreateNote} 
            notes={notes} 
            isLoading={isLoading} 
          />
        )}

        {activeTab === 'search' && (
          <SearchView 
            searchQuery={searchQuery} 
            setSearchQuery={setSearchQuery} 
            handleSearch={handleSearch} 
            memories={memories} 
            isLoading={isLoading} 
          />
        )}

        {activeTab === 'relations' && (
          <RelationsView />
        )}

        {activeTab === 'stats' && (
          <StatsView stats={stats} />
        )}
      </main>
    </div>
  );
};

// Dashboard View Component
const DashboardView = ({ stats, isLoading }: { stats: MemoryStats | null, isLoading: boolean }) => {
  return (
    <div className="dashboard-view">
      <h2>System Dashboard</h2>
      
      <div className="stats-grid">
        <StatCard 
          title="Total Memories" 
          value={stats?.graph_nodes || 0} 
          description="Nodes in the memory graph" 
          icon="🧠" 
        />
        <StatCard 
          title="Connections" 
          value={stats?.graph_edges || 0} 
          description="Relationships between memories" 
          icon="🔗" 
        />
        <StatCard 
          title="Memory Types" 
          value={5} 
          description="Rule, Procedure, Concept, etc." 
          icon="🏷️" 
        />
        <StatCard 
          title="Active Processes" 
          value={3} 
          description="Running memory optimization tasks" 
          icon="⚙️" 
        />
      </div>

      <div className="recent-activity">
        <h3>Recent Activity</h3>
        <ul className="activity-list">
          <li><span className="time">2 min ago</span> - Memory enzyme completed: 2 links pruned, 1 summary refined</li>
          <li><span className="time">5 min ago</span> - New note stored: "Understanding neural networks fundamentals"</li>
          <li><span className="time">12 min ago</span> - Search query: "quantum computing principles" (2 results)</li>
          <li><span className="time">18 min ago</span> - Relation added: "neural-networks" → "deep-learning"</li>
        </ul>
      </div>

      <div className="system-status">
        <h3>System Status</h3>
        <div className="status-indicators">
          <StatusIndicator label="Memory Core" status="operational" />
          <StatusIndicator label="Vector Store" status="operational" />
          <StatusIndicator label="Graph Engine" status="operational" />
          <StatusIndicator label="Enzyme Scheduler" status="running" />
        </div>
      </div>
    </div>
  );
};

// Stat Card Component
const StatCard = ({ title, value, description, icon }: { 
  title: string; 
  value: number; 
  description: string; 
  icon: string; 
}) => (
  <div className="stat-card">
    <div className="icon">{icon}</div>
    <h3>{title}</h3>
    <div className="value">{value}</div>
    <p className="description">{description}</p>
  </div>
);

// Status Indicator Component
const StatusIndicator = ({ label, status }: { label: string; status: string }) => (
  <div className={`status-indicator ${status}`}>
    <span className="label">{label}</span>
    <span className="status">{status}</span>
  </div>
);

// Notes View Component
const NotesView = ({ 
  newNote, 
  setNewNote, 
  handleCreateNote, 
  notes, 
  isLoading 
}: { 
  newNote: string; 
  setNewNote: (value: string) => void; 
  handleCreateNote: () => void; 
  notes: Note[]; 
  isLoading: boolean; 
}) => {
  return (
    <div className="notes-view">
      <h2>Create New Memory Note</h2>
      
      <div className="input-group">
        <textarea
          value={newNote}
          onChange={(e) => setNewNote(e.target.value)}
          placeholder="Enter your memory/note here..."
          rows={4}
        />
        <button 
          onClick={handleCreateNote} 
          disabled={isLoading || !newNote.trim()}
          className="primary-btn"
        >
          {isLoading ? 'Storing...' : 'Store Memory'}
        </button>
      </div>

      <h2>Stored Memories</h2>
      {notes.length === 0 ? (
        <p className="empty-state">No memories stored yet. Create your first memory above!</p>
      ) : (
        <div className="notes-grid">
          {notes.map(note => (
            <NoteCard key={note.id} note={note} />
          ))}
        </div>
      )}
    </div>
  );
};

// Note Card Component
const NoteCard = ({ note }: { note: Note }) => (
  <div className="note-card">
    <div className="note-header">
      <span className={`note-type type-${note.type}`}>{note.type}</span>
      <small>ID: {note.id.substring(0, 8)}...</small>
    </div>
    <h3>{note.summary}</h3>
    <p className="note-content">{note.content}</p>
    <div className="note-meta">
      <div className="keywords">
        {note.keywords.map((kw, idx) => (
          <span key={idx} className="keyword">{kw}</span>
        ))}
      </div>
      <div className="tags">
        {note.tags.map((tag, idx) => (
          <span key={idx} className="tag">{tag}</span>
        ))}
      </div>
    </div>
  </div>
);

// Search View Component
const SearchView = ({ 
  searchQuery, 
  setSearchQuery, 
  handleSearch, 
  memories, 
  isLoading 
}: { 
  searchQuery: string; 
  setSearchQuery: (value: string) => void; 
  handleSearch: () => void; 
  memories: Note[]; 
  isLoading: boolean; 
}) => {
  return (
    <div className="search-view">
      <h2>Search Memories</h2>
      
      <div className="input-group">
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Enter search query..."
        />
        <button 
          onClick={handleSearch} 
          disabled={isLoading || !searchQuery.trim()}
          className="primary-btn"
        >
          {isLoading ? 'Searching...' : 'Find Memories'}
        </button>
      </div>

      <h2>Search Results</h2>
      {memories.length === 0 ? (
        <p className="empty-state">Enter a query above to search your memories</p>
      ) : (
        <div className="results-list">
          {memories.map(memory => (
            <MemoryResult key={memory.id} memory={memory} />
          ))}
        </div>
      )}
    </div>
  );
};

// Memory Result Component
const MemoryResult = ({ memory }: { memory: Note }) => (
  <div className="memory-result">
    <div className="result-header">
      <h3>{memory.summary}</h3>
      {memory.relevance_score && (
        <span className="relevance-score">
          Relevance: {(memory.relevance_score * 100).toFixed(0)}%
        </span>
      )}
    </div>
    <p>{memory.content}</p>
    <div className="result-meta">
      <div className="keywords">
        {memory.keywords.map((kw, idx) => (
          <span key={idx} className="keyword">{kw}</span>
        ))}
      </div>
      <div className="tags">
        {memory.tags.map((tag, idx) => (
          <span key={idx} className="tag">{tag}</span>
        ))}
      </div>
      {memory.connected_memories && (
        <span className="connections">
          {memory.connected_memories} connected memories
        </span>
      )}
    </div>
  </div>
);

// Relations View Component
const RelationsView = () => {
  return (
    <div className="relations-view">
      <h2>Memory Relations</h2>
      <p>Visualize and manage relationships between memories.</p>
      
      <div className="relations-controls">
        <div className="input-group">
          <input type="text" placeholder="Source Memory ID" />
          <input type="text" placeholder="Target Memory ID" />
          <select>
            <option value="relates_to">Relates To</option>
            <option value="causes">Causes</option>
            <option value="similar_to">Similar To</option>
            <option value="part_of">Part Of</option>
          </select>
          <button className="primary-btn">Add Relation</button>
        </div>
      </div>

      <div className="relations-graph-placeholder">
        <h3>Memory Graph Visualization</h3>
        <p>This would display an interactive graph showing relationships between memories.</p>
        <div className="graph-preview">
          <div className="node">Concept A</div>
          <div className="relation">relates_to</div>
          <div className="node">Concept B</div>
          <div className="relation">causes</div>
          <div className="node">Outcome C</div>
        </div>
      </div>
    </div>
  );
};

// Stats View Component
const StatsView = ({ stats }: { stats: MemoryStats | null }) => {
  return (
    <div className="stats-view">
      <h2>System Statistics</h2>
      
      {stats ? (
        <div className="stats-overview">
          <div className="stat-item">
            <h3>Memory Graph</h3>
            <p><strong>Nodes:</strong> {stats.graph_nodes}</p>
            <p><strong>Edges:</strong> {stats.graph_edges}</p>
          </div>
          
          <div className="stat-item">
            <h3>Storage</h3>
            <p><strong>Total Memories:</strong> {stats.memory_count}</p>
            <p><strong>Connections:</strong> {stats.connection_count}</p>
          </div>
        </div>
      ) : (
        <p>Loading statistics...</p>
      )}

      <div className="performance-metrics">
        <h3>Performance Metrics</h3>
        <div className="metric">
          <span>Response Time:</span>
          <span className="value">42ms avg</span>
        </div>
        <div className="metric">
          <span>Memory Usage:</span>
          <span className="value">128MB</span>
        </div>
        <div className="metric">
          <span>Cache Hit Rate:</span>
          <span className="value">94%</span>
        </div>
      </div>

      <div className="maintenance-actions">
        <h3>Maintenance Actions</h3>
        <div className="action-buttons">
          <button className="secondary-btn">Run Memory Enzymes</button>
          <button className="secondary-btn">Optimize Storage</button>
          <button className="warning-btn">Reset Memory</button>
        </div>
      </div>
    </div>
  );
};

export default App;
