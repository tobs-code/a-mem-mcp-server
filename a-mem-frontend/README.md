# A-MEM MCP Server Frontend

A beautiful and functional frontend for the A-MEM (Agentic Memory System) MCP (Model Context Protocol) Server. This interface allows users to interact with the memory management system through an intuitive web interface.

## Features

- **Dashboard**: Overview of memory system statistics and recent activity
- **Memory Notes**: Create, view, and manage atomic memory notes
- **Search**: Semantic search across all stored memories
- **Relations**: Visualize and manage relationships between memories
- **System Stats**: Monitor memory graph metrics and performance

## Architecture

This frontend communicates with the A-MEM MCP Server through simulated API calls that mirror the actual MCP tool calls. The interface demonstrates how various MCP tools can be utilized:

- `create_atomic_note`: Store new pieces of information
- `retrieve_memories`: Search for relevant memories based on semantic similarity
- `get_memory_stats`: Retrieve statistics about the memory system
- `list_notes`: List all stored notes from the memory graph
- `get_note`: Retrieve a single note by ID
- `delete_atomic_note`: Delete a note from the memory system
- `get_graph`: Retrieve the memory graph structure
- `run_memory_enzymes`: Execute memory optimization processes
- `research_and_store`: Perform web research and store findings

## Development

This is a React application built with Vite and TypeScript:

- **Framework**: React 18+
- **Build Tool**: Vite
- **Language**: TypeScript
- **Styling**: CSS Modules

### Prerequisites

- Node.js 16+ 
- npm or yarn

### Running the Application

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

3. Open your browser to the displayed URL (typically http://localhost:5173)

### Building for Production

```bash
npm run build
```

## Integration with A-MEM MCP Server

The frontend simulates communication with the A-MEM MCP Server. In a production setup, this would connect to the actual MCP server which provides tools for:

- Memory storage and retrieval
- Semantic search capabilities
- Relationship mapping between memories
- Automated memory optimization
- Web research and content extraction

## Components

The application is organized into several key components:

- **Dashboard**: System overview with metrics and recent activity
- **Notes View**: Interface for creating and viewing memory notes
- **Search View**: Query interface for finding relevant memories
- **Relations View**: Tool for managing connections between memories
- **Stats View**: Detailed system statistics and maintenance actions

## Design Philosophy

The interface follows modern UI/UX principles with:

- Clean, accessible design
- Responsive layout for different screen sizes
- Intuitive navigation between features
- Clear visualization of complex memory relationships
- Performance optimized components

## MCP Server Capabilities

The A-MEM MCP Server provides sophisticated memory management through:

- **Atomic Note Creation**: Automatically classify and link new information
- **Semantic Search**: Find relevant memories using vector similarity
- **Relationship Mapping**: Understand connections between concepts
- **Memory Enzymes**: Optimize and refine memory structures over time
- **Web Research**: Automatically gather and integrate external information
- **Evolution Process**: Continuously improve memory organization

## License

This project is part of the A-MEM ecosystem. See the main project repository for licensing information.
