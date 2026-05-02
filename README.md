# RAG AI Chatbot

A production-quality Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and ask questions about their content using local LLMs.

## Features

- **Document Upload**: Upload PDF files for processing
- **Intelligent Chunking**: Automatically splits documents into optimal chunks
- **Semantic Search**: Uses sentence-transformers for accurate similarity search
- **Local LLM Integration**: Uses Ollama with Mistral model for responses
- **Source Attribution**: Shows source chunks used for each answer
- **Modern UI**: Clean, responsive dark-themed interface
- **Real-time Chat**: Interactive chat interface with typing indicators

## Architecture

### Backend (FastAPI)
- **main.py**: API routes and server setup
- **rag_pipeline.py**: Complete RAG orchestration
- **embeddings.py**: Sentence-transformer embeddings
- **vectorstore.py**: FAISS vector database
- **utils.py**: PDF processing and text chunking

### Frontend
- **index.html**: Single-page application with embedded CSS/JavaScript

## Prerequisites

1. **Python 3.8+**
2. **Ollama** (for local LLM)
3. **Git** (optional, for cloning)

## Installation

### 1. Install Ollama

```bash
# Install Ollama (Linux/macOS)
curl -fsSL https://ollama.ai/install.sh | sh

# For Windows, download from: https://ollama.ai/download

# Start Ollama service
ollama serve
```

### 2. Pull the Mistral Model

```bash
ollama pull mistral
```

### 3. Set Up the Project

```bash
# Clone or download the project
cd "Rag ai CHATBOT"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Application

### 1. Start the Backend Server

```bash
cd backend
python main.py
```

The server will start on `http://localhost:8000`

### 2. Open the Frontend

Open `frontend/index.html` in your web browser, or simply double-click the file.

Alternatively, you can serve it with a simple HTTP server:

```bash
cd frontend
python -m http.server 3000
```

Then open `http://localhost:3000`

## Usage

### 1. Upload Documents
- Click the upload area or drag and drop PDF files
- Wait for processing confirmation
- Check the document count in the sidebar

### 2. Ask Questions
- Type your question in the chat input
- Press Enter or click the send button
- View the answer with source citations

### 3. Manage Documents
- Use "Clear Documents" to reset the system
- Upload new documents as needed

## API Endpoints

### Upload Document
```http
POST /upload
Content-Type: multipart/form-data

file: PDF file
```

### Chat Query
```http
POST /chat
Content-Type: application/json

{
  "query": "Your question here",
  "k": 3  // Number of sources to retrieve (optional)
}
```

### Health Check
```http
GET /health
```

### Document Count
```http
GET /documents/count
```

### Clear Documents
```http
DELETE /documents
```

## Configuration

### Environment Variables (Optional)

Create a `.env` file in the backend directory:

```env
# Ollama Configuration
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=mistral

# Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Chunking Parameters
CHUNK_SIZE=500
CHUNK_OVERLAP=100

# Server Configuration
HOST=0.0.0.0
PORT=8000
```

### Customization

#### Change LLM Model
Edit `backend/rag_pipeline.py`:
```python
self.llm_model = "llama2"  # or any other Ollama model
```

#### Change Embedding Model
Edit `backend/rag_pipeline.py`:
```python
self.embedding_model = "all-mpnet-base-v2"  # or other sentence-transformer model
```

#### Adjust Chunking Parameters
Edit `backend/rag_pipeline.py`:
```python
self.chunk_size = 800  # Larger chunks
self.chunk_overlap = 200  # More overlap
```

## Troubleshooting

### Common Issues

#### 1. Ollama Connection Failed
```bash
# Check if Ollama is running
ollama list

# Restart Ollama
ollama serve
```

#### 2. Model Not Found
```bash
# Pull the required model
ollama pull mistral

# Check available models
ollama list
```

#### 3. PDF Processing Errors
- Ensure PDF files are not password-protected
- Check if PDF contains extractable text
- Try with a different PDF file

#### 4. Memory Issues
- Reduce chunk size in configuration
- Use smaller embedding models
- Close other applications

#### 5. Port Already in Use
```bash
# Kill process using port 8000 (Windows)
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Kill process using port 8000 (Linux/macOS)
lsof -ti:8000 | xargs kill -9
```

### Performance Optimization

#### GPU Acceleration
If you have an NVIDIA GPU:

1. Install CUDA-enabled PyTorch:
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

2. Use FAISS GPU:
```bash
pip uninstall faiss-cpu
pip install faiss-gpu
```

#### Memory Optimization
- Reduce `CHUNK_SIZE` in configuration
- Use smaller embedding models
- Clear documents periodically

## Development

### Running Tests
```bash
# Test individual components
python backend/embeddings.py
python backend/vectorstore.py
python backend/rag_pipeline.py
```

### Code Structure
```
backend/
├── main.py              # FastAPI application and routes
├── rag_pipeline.py      # RAG orchestration logic
├── embeddings.py        # Sentence-transformer embeddings
├── vectorstore.py       # FAISS vector database
└── utils.py             # PDF processing utilities

frontend/
└── index.html           # Complete frontend application
```

### Adding New Features

#### New Document Types
Extend `utils.py` to support other formats:
```python
def extract_text_from_docx(file_path: str) -> str:
    # Add DOCX support
    pass
```

#### Custom Embedding Models
Add new models to `embeddings.py`:
```python
def get_custom_embeddings(texts: List[str]) -> np.ndarray:
    # Add custom embedding logic
    pass
```

## Security Considerations

- The API runs without authentication (development setup)
- Documents are processed in memory only
- No data is persisted to disk by default
- Consider adding authentication for production use

## Production Deployment

### Docker Setup
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY backend/ ./backend/
COPY frontend/ ./frontend/

EXPOSE 8000
CMD ["python", "backend/main.py"]
```

### Environment Variables
- Set proper CORS origins in production
- Use environment variables for configuration
- Add authentication and rate limiting
- Consider using reverse proxy (nginx)

## License

This project is open source and available under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

If you encounter any issues:

1. Check the troubleshooting section
2. Verify Ollama is running and the model is available
3. Check the backend logs for error messages
4. Ensure all dependencies are properly installed

## Performance Benchmarks

Typical performance metrics:
- **PDF Processing**: ~2-5 seconds per 10 pages
- **Query Response**: ~3-10 seconds depending on context size
- **Memory Usage**: ~500MB for 100 document chunks
- **Startup Time**: ~10-15 seconds (model loading)

## Future Enhancements

- [ ] Support for more document formats (DOCX, TXT, MD)
- [ ] Conversation history and context
- [ ] Document summarization
- [ ] Multi-language support
- [ ] Advanced search filters
- [ ] Export chat conversations
- [ ] Document management interface
- [ ] User authentication and sessions
