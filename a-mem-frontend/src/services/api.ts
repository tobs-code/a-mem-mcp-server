// api.ts - API service for a-mem-mcp-server communication

// Define TypeScript interfaces for our data structures
export interface Note {
  id: string;
  content: string;
  summary: string;
  keywords: string[];
  tags: string[];
  type: string;
  relevance_score?: number;
  connected_memories?: number;
  connected_context?: string;
}

export interface MemoryStats {
  status: string;
  graph_nodes: number;
  graph_edges: number;
  memory_count: number;
  connection_count: number;
}

export interface ToolCallRequest {
  method: string;
  params: Record<string, any>;
}

export interface ToolCallResponse {
  result?: any;
  error?: {
    code: number;
    message: string;
  };
}

class AMemAPIService {
  // Store for mock data during development
  private mockNotes: Note[] = [];
  private mockStats: MemoryStats = {
    status: 'success',
    graph_nodes: 0,
    graph_edges: 0,
    memory_count: 0,
    connection_count: 0
  };

  constructor() {
    // Initialize with some mock data for development
    this.initializeMockData();
  }

  private initializeMockData() {
    this.mockStats = {
      status: 'success',
      graph_nodes: 42,
      graph_edges: 128,
      memory_count: 42,
      connection_count: 128
    };

    this.mockNotes = [
      {
        id: 'note-1',
        content: 'Neural networks are composed of layers of interconnected nodes that process data through weighted connections.',
        summary: 'Fundamentals of neural network architecture',
        keywords: ['neural networks', 'machine learning', 'artificial intelligence'],
        tags: ['concept', 'technical'],
        type: 'concept',
        relevance_score: 0.95,
        connected_memories: 3,
        connected_context: 'Related to deep learning, backpropagation, activation functions'
      },
      {
        id: 'note-2',
        content: 'Quantum computing leverages quantum bits (qubits) that can exist in superposition states simultaneously.',
        summary: 'Introduction to quantum computing principles',
        keywords: ['quantum computing', 'qubits', 'superposition'],
        tags: ['concept', 'physics'],
        type: 'concept',
        relevance_score: 0.87,
        connected_memories: 2,
        connected_context: 'Related to quantum entanglement, quantum algorithms'
      },
      {
        id: 'note-3',
        content: 'The Transformer architecture revolutionized NLP by using attention mechanisms instead of sequential processing.',
        summary: 'Transformer model architecture',
        keywords: ['transformers', 'attention mechanism', 'nlp'],
        tags: ['model', 'architecture'],
        type: 'concept',
        relevance_score: 0.92,
        connected_memories: 4,
        connected_context: 'Related to BERT, GPT, self-attention'
      }
    ];
  }

  // Method to simulate calling MCP tools
  async callMCPTool(toolName: string, params: Record<string, any>): Promise<any> {
    // In a real implementation, this would communicate with the MCP server
    // For now, we'll simulate the responses based on the tool name
    
    switch(toolName) {
      case 'create_atomic_note':
        return this.simulateCreateNote(params);
      
      case 'retrieve_memories':
        return this.simulateRetrieveMemories(params);
      
      case 'get_memory_stats':
        return this.simulateGetMemoryStats();
      
      case 'list_notes':
        return this.simulateListNotes();
      
      case 'get_note':
        return this.simulateGetNote(params);
      
      case 'delete_atomic_note':
        return this.simulateDeleteNote(params);
      
      case 'get_graph':
        return this.simulateGetGraph(params);
      
      case 'run_memory_enzymes':
        return this.simulateRunMemoryEnzymes(params);
      
      case 'research_and_store':
        return this.simulateResearchAndStore(params);
      
      default:
        throw new Error(`Unknown tool: ${toolName}`);
    }
  }

  private simulateCreateNote(params: Record<string, any>): Promise<any> {
    return new Promise((resolve) => {
      setTimeout(() => {
        const newNote: Note = {
          id: `note-${Date.now()}`,
          content: params.content || '',
          summary: `Summary of: ${(params.content || '').substring(0, 50)}...`,
          keywords: params.keywords || ['new', 'memory'],
          tags: params.tags || ['user-input'],
          type: params.type || 'concept',
          relevance_score: 0.8,
          connected_memories: 0,
          connected_context: ''
        };
        
        this.mockNotes.push(newNote);
        
        resolve({
          status: 'success',
          note_id: newNote.id,
          message: `Note created with ID: ${newNote.id}. Evolution started in background.`
        });
      }, 500);
    });
  }

  private simulateRetrieveMemories(params: Record<string, any>): Promise<any> {
    return new Promise((resolve) => {
      setTimeout(() => {
        const query = params.query || '';
        const maxResults = params.max_results || 5;
        
        // Simple search simulation
        const results = this.mockNotes
          .filter(note => 
            note.content.toLowerCase().includes(query.toLowerCase()) ||
            note.summary.toLowerCase().includes(query.toLowerCase()) ||
            note.keywords.some(kw => kw.toLowerCase().includes(query.toLowerCase()))
          )
          .slice(0, maxResults)
          .map(note => ({
            ...note,
            relevance_score: Math.random() * 0.5 + 0.5 // Random score between 0.5 and 1.0
          }));
        
        resolve({
          status: 'success',
          query: query,
          results_count: results.length,
          results: results
        });
      }, 700);
    });
  }

  private simulateGetMemoryStats(): Promise<MemoryStats> {
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve(this.mockStats);
      }, 300);
    });
  }

  private simulateListNotes(): Promise<{ notes: Note[] }> {
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve({ notes: [...this.mockNotes] });
      }, 400);
    });
  }

  private simulateGetNote(params: Record<string, any>): Promise<any> {
    return new Promise((resolve, reject) => {
      setTimeout(() => {
        const noteId = params.note_id;
        const note = this.mockNotes.find(n => n.id === noteId);
        
        if (note) {
          resolve({ note });
        } else {
          reject({ error: `Note '${noteId}' not found` });
        }
      }, 300);
    });
  }

  private simulateDeleteNote(params: Record<string, any>): Promise<any> {
    return new Promise((resolve, reject) => {
      setTimeout(() => {
        const noteId = params.note_id;
        const initialLength = this.mockNotes.length;
        this.mockNotes = this.mockNotes.filter(n => n.id !== noteId);
        
        if (this.mockNotes.length < initialLength) {
          resolve({
            status: 'success',
            message: `Note ${noteId} deleted successfully`
          });
        } else {
          reject({ error: `Note '${noteId}' not found` });
        }
      }, 400);
    });
  }

  private simulateGetGraph(params: Record<string, any>): Promise<any> {
    return new Promise((resolve) => {
      setTimeout(() => {
        // Generate a simple graph representation
        const nodes = this.mockNotes.map(note => ({
          id: note.id,
          label: note.summary.substring(0, 30) + '...',
          type: note.type,
          keywords: note.keywords.slice(0, 3),
          relevance: note.relevance_score || 0.5
        }));

        // Create some random connections between notes
        const edges = [];
        for (let i = 0; i < nodes.length - 1; i++) {
          if (Math.random() > 0.3) { // 70% chance of connection
            edges.push({
              source: nodes[i].id,
              target: nodes[i + 1].id,
              type: 'related_to',
              weight: Math.random()
            });
          }
        }

        resolve({
          nodes,
          edges,
          saved_to_disk: params.save || false,
          memory_nodes: nodes.length,
          memory_edges: edges.length
        });
      }, 600);
    });
  }

  private simulateRunMemoryEnzymes(params: Record<string, any>): Promise<any> {
    return new Promise((resolve) => {
      setTimeout(() => {
        // Simulate enzyme actions
        const prunedCount = Math.floor(Math.random() * 5);
        const suggestionsCount = Math.floor(Math.random() * 3);
        const summariesRefined = Math.floor(Math.random() * 2);
        
        resolve({
          status: 'success',
          results: {
            pruned_count: prunedCount,
            suggestions_count: suggestionsCount,
            summaries_refined: summariesRefined,
            digested_count: Math.floor(Math.random() * 3),
            zombie_nodes_removed: Math.floor(Math.random() * 2),
            suggestions: []
          },
          message: `Enzymes completed: ${prunedCount} links pruned, ${suggestionsCount} relations suggested, ${summariesRefined} summaries refined.`,
          suggested_relations: []
        });
      }, 1000);
    });
  }

  private simulateResearchAndStore(params: Record<string, any>): Promise<any> {
    return new Promise((resolve) => {
      setTimeout(() => {
        const query = params.query || 'general topic';
        const maxSources = params.max_sources || 1;
        
        // Simulate research results
        const researchNotes = Array.from({ length: maxSources }, (_, i) => ({
          id: `research-note-${Date.now()}-${i}`,
          content: `Research result for "${query}" - finding #${i + 1}. This is simulated content from web research.`,
          summary: `Research finding related to "${query}"`,
          keywords: [query, 'research', 'web-search'],
          tags: ['research', 'external'],
          type: 'reference',
          relevance_score: 0.8,
          connected_memories: 0,
          connected_context: ''
        }));
        
        // Add to our mock store
        this.mockNotes.push(...researchNotes);
        
        resolve({
          status: 'success',
          message: `Research completed: ${researchNotes.length} notes stored`,
          query: query,
          notes_created: researchNotes.length,
          notes_stored: researchNotes.length,
          notes_failed: 0,
          stored_notes: researchNotes.map(note => ({
            id: note.id,
            summary: note.summary,
            type: note.type,
            source_url: 'https://example.com/simulated-source'
          })),
          failed_notes: null
        });
      }, 1500);
    });
  }
}

export const apiService = new AMemAPIService();

// Export individual convenience methods
export const createNote = (content: string, source: string = 'user_input') => 
  apiService.callMCPTool('create_atomic_note', { content, source });

export const retrieveMemories = (query: string, maxResults: number = 5) =>
  apiService.callMCPTool('retrieve_memories', { query, max_results: maxResults });

export const getMemoryStats = () => 
  apiService.callMCPTool('get_memory_stats', {});

export const listNotes = () => 
  apiService.callMCPTool('list_notes', {});

export const getNote = (noteId: string) => 
  apiService.callMCPTool('get_note', { note_id: noteId });

export const deleteNote = (noteId: string) => 
  apiService.callMCPTool('delete_atomic_note', { note_id: noteId });

export const getGraph = (saveToDisk: boolean = false) => 
  apiService.callMCPTool('get_graph', { save: saveToDisk });

export const runMemoryEnzymes = (params: Record<string, any> = {}) => 
  apiService.callMCPTool('run_memory_enzymes', params);

export const researchAndStore = (query: string, maxSources: number = 1) => 
  apiService.callMCPTool('research_and_store', { query, max_sources: maxSources });