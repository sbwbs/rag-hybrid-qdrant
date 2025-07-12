# Hybrid Search RFP Assistant

A Streamlit-based application that provides a hybrid search interface for RFP (Request for Proposal) documents using dense and sparse embeddings. The application combines the power of OpenAI's embeddings, Qdrant's vector database, and GPT models to provide accurate and context-aware answers to RFP questions.

## Features

- **Hybrid Search**: Combines dense and sparse embeddings for improved search accuracy
- **Multiple Document Formats**: Support for both JSON and CSV document uploads
- **CSV Document Upload**: Direct CSV upload for Q&A pairs - no manual conversion needed
- **Bulk Question Processing**: Process multiple questions at once and get batch answers
- **Intelligent Answering**: Uses GPT models to generate context-aware answers
- **Confidence Scoring**: Provides confidence scores with detailed breakdowns
- **Real-time Search**: Instant results with relevance scores
- **Template Downloads**: Get properly formatted templates for easy uploads
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **User-friendly Interface**: Clean and intuitive Streamlit UI with clear guidance

## Architecture

The application consists of three main components:

1. **Search Engine** (`search_engine.py`):
   - Handles document indexing and search operations
   - Combines dense (OpenAI) and sparse (FastEmbed) embeddings
   - Manages Qdrant vector database operations
   - Generates answers using GPT models

2. **Document Processor** (`document_processor.py`):
   - Validates and cleans document input
   - Ensures data consistency and quality
   - Handles bulk document processing

3. **Web Interface** (`app.py`):
   - Streamlit-based user interface
   - Document management and search interface
   - Settings configuration
   - Real-time search results display

## Prerequisites

- Python 3.8+
- OpenAI API key
- Qdrant instance (self-hosted or cloud)
- Qdrant API key

## Installation

1. Clone the repository:
```bash
git clone git@github.com:sbwbs/rag-hybrid-qdrant.git
cd hybrid-search-rfp-assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with the following variables:
```env
OPENAI_API_KEY=your_openai_api_key
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_URL=your_qdrant_url
COLLECTION_NAME=name_of_your_collection
LLM_MODEL=gpt-4o
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Access the application in your web browser at `http://localhost:8501`

3. Navigate through the different sections:
   - **Search**: Enter questions and get answers from indexed documents
   - **Document Upload**: Upload JSON files with structured Q&A documents
   - **CSV Document Upload**: Upload CSV files with question/answer pairs
   - **Bulk Processing**: Process multiple questions at once to get batch answers
   - **Settings**: Configure API keys and other settings

## Document Formats

The application supports two document upload formats:

### JSON Format
Upload structured documents in JSON format:
```json
{
  "documents": [
    {
      "question": "What is the company's approach to data security?",
      "answer": "Our company implements a multi-layered security approach...",
      "summary": "Overview of data security measures",
      "answer_type": "security",
      "date": "2024-01-01"
    }
  ]
}
```

### CSV Format
Upload Q&A pairs directly in CSV format:
```csv
question,answer,summary,answer_type,date
"What is our data security approach?","We implement multi-layered security...","Security overview","security","2024-01-01"
"How do we handle customer privacy?","We follow GDPR compliance standards...","Privacy policy","privacy","2024-01-01"
```

**Required columns:** `question`, `answer`  
**Optional columns:** `summary`, `answer_type`, `date`

## Bulk Question Processing

Process multiple questions at once using CSV or Excel files:

**Input Format:**
```csv
question
"What is our refund policy?"
"How do we handle customer complaints?"
"What are our payment terms?"
```

**Output:** Downloadable file with questions, AI-generated answers, confidence scores, and source documents.

**Features:**
- Upload CSV/Excel with up to 1,000 questions
- Parallel processing for faster results
- Template download for correct format
- Comprehensive results with confidence metrics

## Logging

The application includes comprehensive logging:
- Logs are stored in the `logs` directory
- Separate log files for each component
- Log rotation (10MB max size, 5 backup files)
- Different log levels (DEBUG, INFO, WARNING, ERROR)
- Detailed error tracking and debugging information

## Error Handling

The application includes robust error handling:
- Input validation
- API error handling
- User-friendly error messages
- Detailed error logging
- Graceful degradation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Acknowledgments

- OpenAI for embeddings and GPT models
- Qdrant for vector database
- FastEmbed for sparse embeddings
- Streamlit for the web interface
