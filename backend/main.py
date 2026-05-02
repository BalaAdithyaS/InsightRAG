"""
FastAPI Backend for RAG AI Chatbot
Main API endpoints for document upload and chat functionality
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import logging
import os
from pathlib import Path

from rag_pipeline import RAGPipeline
from utils import validate_pdf_file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG AI Chatbot",
    description="A chatbot that answers questions based on uploaded PDF documents",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG pipeline
rag_pipeline = RAGPipeline()

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Pydantic models
class ChatRequest(BaseModel):
    query: str
    k: Optional[int] = 3  # Number of top-k chunks to retrieve

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    query: str

class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_processed: int

# Health check endpoint
@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "RAG AI Chatbot API is running"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "components": {
            "ollama": "connected",  # Will be verified during actual requests
            "vectorstore": "initialized",
            "embeddings": "ready"
        }
    }

# Upload PDF endpoint
@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file and process it for RAG
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400, 
                detail="Only PDF files are allowed"
            )
        
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"Processing uploaded file: {file.filename}")
        
        # Process PDF through RAG pipeline
        chunks_processed = await rag_pipeline.process_document(str(file_path))
        
        # Clean up uploaded file
        os.remove(file_path)
        
        return UploadResponse(
            message=f"Successfully processed {file.filename}",
            filename=file.filename,
            chunks_processed=chunks_processed
        )
        
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        # Clean up file if it exists
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )

# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Ask a question and get answer with sources
    """
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Check if any documents have been processed
        if not rag_pipeline.has_documents():
            raise HTTPException(
                status_code=400,
                detail="No documents have been uploaded. Please upload a PDF first."
            )
        
        # Get response from RAG pipeline
        answer, sources = await rag_pipeline.query(
            query=request.query,
            k=request.k
        )
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            query=request.query
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

# Clear documents endpoint
@app.delete("/documents")
async def clear_documents():
    """
    Clear all processed documents from the vector store
    """
    try:
        rag_pipeline.clear_documents()
        return {"message": "All documents cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing documents: {str(e)}"
        )

# Get document count endpoint
@app.get("/documents/count")
async def get_document_count():
    """
    Get the number of processed document chunks
    """
    try:
        count = rag_pipeline.get_document_count()
        return {"document_chunks": count}
    except Exception as e:
        logger.error(f"Error getting document count: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting document count: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
