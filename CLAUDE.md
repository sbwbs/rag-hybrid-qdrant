# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
streamlit run app.py
```
Access the application at `http://localhost:8501`

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Environment Setup
Create a `.env` file in the root directory with:
```env
OPENAI_API_KEY=your_openai_api_key
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_URL=your_qdrant_url
COLLECTION_NAME=hybrid_rfp_rag
LLM_MODEL=gpt-4o
SEARCH_TOP_K=5
```

## Architecture Overview

This is a **hybrid search RFP assistant** that combines dense and sparse embeddings for document retrieval and answer generation. The application uses:

- **Dense embeddings**: OpenAI's `text-embedding-3-small` model for semantic search
- **Sparse embeddings**: FastEmbed's Splade model for keyword-based search  
- **Vector database**: Qdrant for storing and searching embeddings
- **LLM**: GPT models for answer generation
- **UI**: Streamlit web interface

### Core Components

1. **HybridSearchEngine** (`search_engine.py`):
   - Manages dual embedding generation (dense + sparse)
   - Handles Qdrant collection setup and operations
   - Performs fusion search combining both embedding types
   - Generates contextual answers using retrieved documents

2. **DocumentProcessor** (`document_processor.py`):
   - Validates document structure and required fields
   - Processes and cleans document data
   - Ensures data consistency before indexing

3. **BulkProcessor** (`bulk_processor.py`):
   - Handles batch processing of multiple questions
   - Supports CSV/Excel input files
   - Parallel processing with configurable worker threads
   - Export results in multiple formats

4. **Streamlit App** (`app.py`):
   - Multi-page interface: Search, Document Upload, Bulk Processing, Settings
   - Session state management for component initialization
   - File upload and processing workflows

5. **Configuration** (`config.py`):
   - Environment variable management
   - API key and endpoint configuration
   - Search parameter defaults

### Data Flow

1. **Document Indexing**: Documents → DocumentProcessor → dual embeddings → Qdrant storage
2. **Search**: Query → dual embeddings → Qdrant fusion search → context retrieval → LLM answer generation
3. **Bulk Processing**: CSV/Excel → question extraction → parallel search → results export

### Document Structure

Documents must have this JSON structure:
```json
{
  "documents": [
    {
      "question": "string (required)",
      "answer": "string (required)", 
      "summary": "string (optional)",
      "answer_type": "string (optional)",
      "date": "string (optional)"
    }
  ]
}
```

### Key Dependencies

- `streamlit`: Web interface framework
- `qdrant-client`: Vector database client
- `openai`: OpenAI API client for embeddings and LLM
- `fastembed`: Sparse embedding generation
- `python-dotenv`: Environment variable management
- `pandas`: Data processing for bulk operations

### Logging

- Comprehensive logging via `logging_config.py`
- Separate loggers for each component (app, search_engine, document_processor, bulk_processor)
- Log files stored in `logs/` directory with rotation (10MB max, 5 backups)
- Log levels: DEBUG, INFO, WARNING, ERROR

### Session State Management

The Streamlit app maintains these session state objects:
- `config`: Configuration instance
- `search_engine`: HybridSearchEngine instance  
- `document_processor`: DocumentProcessor instance
- `bulk_processor`: BulkProcessor instance

Components are initialized once and cached using `@st.cache_resource` decorators.