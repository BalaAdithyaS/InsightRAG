"""
RAG (Retrieval-Augmented Generation) Pipeline
Handles the complete flow from document processing to query answering
"""

import logging
from typing import List, Dict, Tuple, Optional
import requests
import json
import time

from utils import process_pdf_document
from embeddings import get_embedding_generator
from vectorstore import get_vector_store

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Complete RAG pipeline for document processing and query answering
    """
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 ollama_url: str = "http://localhost:11434",
                 llm_model: str = "mistral",
                 chunk_size: int = 500,
                 chunk_overlap: int = 100):
        """
        Initialize the RAG pipeline
        
        Args:
            embedding_model: Name of the sentence-transformer model
            ollama_url: URL for Ollama API
            llm_model: Name of the LLM model in Ollama
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.embedding_model = embedding_model
        self.ollama_url = ollama_url
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self.embedding_generator = None
        self.vector_store = None
        
        # Initialize pipeline
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize all pipeline components"""
        try:
            # Initialize embedding generator
            logger.info("Initializing embedding generator...")
            self.embedding_generator = get_embedding_generator(self.embedding_model)
            
            # Initialize vector store
            embedding_dim = self.embedding_generator.get_embedding_dimension()
            logger.info(f"Initializing vector store with dimension {embedding_dim}...")
            self.vector_store = get_vector_store(embedding_dim, index_type="flat")
            
            # Test Ollama connection
            self._test_ollama_connection()
            
            logger.info("RAG pipeline initialized successfully!")
            
        except Exception as e:
            logger.error(f"Error initializing RAG pipeline: {str(e)}")
            raise Exception(f"Failed to initialize RAG pipeline: {str(e)}")
    
    def _test_ollama_connection(self):
        """Test connection to Ollama API"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info(f"Successfully connected to Ollama at {self.ollama_url}")
                
                # Check if model is available
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                
                if self.llm_model not in model_names:
                    logger.warning(f"Model '{self.llm_model}' not found in available models: {model_names}")
                    logger.info(f"You may need to run: ollama pull {self.llm_model}")
                else:
                    logger.info(f"Model '{self.llm_model}' is available")
            else:
                raise Exception(f"Ollama API returned status {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to Ollama: {str(e)}")
            raise Exception(f"Failed to connect to Ollama at {self.ollama_url}. Make sure Ollama is running.")
    
    async def process_document(self, file_path: str) -> int:
        """
        Process a PDF document and add it to the vector store
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Number of chunks processed
        """
        try:
            logger.info(f"Processing document: {file_path}")
            
            # Process PDF and get chunks
            chunk_data = process_pdf_document(
                file_path, 
                self.chunk_size, 
                self.chunk_overlap
            )
            
            if not chunk_data:
                raise ValueError("No chunks were created from the document")
            
            # Extract texts for embedding
            texts = [chunk['text'] for chunk in chunk_data]
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} chunks...")
            embeddings = self.embedding_generator.encode(texts, show_progress=True)
            
            # Add to vector store
            logger.info("Adding chunks to vector store...")
            self.vector_store.add_documents(embeddings, chunk_data)
            
            logger.info(f"Successfully processed document: {len(chunk_data)} chunks")
            return len(chunk_data)
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise Exception(f"Failed to process document: {str(e)}")
    
    async def query(self, query: str, k: int = 3) -> Tuple[str, List[Dict]]:
        """
        Process a query and return answer with sources
        
        Args:
            query: User query
            k: Number of top chunks to retrieve
            
        Returns:
            Tuple of (answer, source_documents)
        """
        try:
            logger.info(f"Processing query: {query}")
            
            # Generate query embedding
            query_embedding = self.embedding_generator.encode_single(query)
            
            # Retrieve relevant documents
            logger.info(f"Retrieving top {k} relevant chunks...")
            retrieved_docs, similarity_scores = self.vector_store.search(query_embedding, k)
            
            if not retrieved_docs:
                return "I couldn't find any relevant information in the uploaded documents to answer your question.", []
            
            # Construct prompt with retrieved context
            context = self._construct_context(retrieved_docs)
            prompt = self._construct_prompt(query, context)
            
            # Generate answer using LLM
            logger.info("Generating answer with LLM...")
            answer = await self._generate_answer(prompt)
            
            # Prepare source information
            sources = self._prepare_sources(retrieved_docs, similarity_scores)
            
            logger.info(f"Query processed successfully. Sources: {len(sources)}")
            return answer, sources
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise Exception(f"Failed to process query: {str(e)}")
    
    def _construct_context(self, retrieved_docs: List[Dict]) -> str:
        """
        Construct context string from retrieved documents
        
        Args:
            retrieved_docs: List of retrieved document chunks
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs):
            context_part = f"""
Document {i+1} (Source: {doc['source']}, Similarity: {doc.get('similarity_score', 0):.3f}):
{doc['text']}
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _construct_prompt(self, query: str, context: str) -> str:
        """
        Construct the prompt for the LLM
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Complete prompt for LLM
        """
        prompt = f"""You are a helpful AI assistant that answers questions based on the provided context. 
Please answer the user's question using only the information from the context below.
If the context doesn't contain enough information to answer the question, say so clearly.
Be accurate, concise, and provide a complete answer.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
        
        return prompt
    
    async def _generate_answer(self, prompt: str) -> str:
        """
        Generate answer using Ollama API
        
        Args:
            prompt: Complete prompt for the LLM
            
        Returns:
            Generated answer
        """
        try:
            # Prepare request payload
            payload = {
                "model": self.llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for more factual answers
                    "max_tokens": 1000,  # Reasonable limit for answers
                    "top_p": 0.9,
                    "top_k": 40
                }
            }
            
            # Make request to Ollama
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=120  # 120 second timeout for first model load
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                
                if not answer:
                    raise Exception("Empty response from LLM")
                
                return answer
            else:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            raise Exception("Request to Ollama timed out. Please try again.")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error communicating with Ollama: {str(e)}")
    
    def _prepare_sources(self, retrieved_docs: List[Dict], similarity_scores: List[float]) -> List[Dict]:
        """
        Prepare source information for the response
        
        Args:
            retrieved_docs: List of retrieved documents
            similarity_scores: Corresponding similarity scores
            
        Returns:
            List of source information
        """
        sources = []
        
        for i, doc in enumerate(retrieved_docs):
            source_info = {
                'source': doc['source'],
                'chunk_index': doc['chunk_index'],
                'similarity_score': float(similarity_scores[i]) if i < len(similarity_scores) else 0.0,
                'text': doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text'],
                'metadata': doc.get('metadata', {})
            }
            sources.append(source_info)
        
        return sources
    
    def has_documents(self) -> bool:
        """Check if any documents have been processed"""
        return self.vector_store.get_document_count() > 0
    
    def get_document_count(self) -> int:
        """Get the number of processed document chunks"""
        return self.vector_store.get_document_count()
    
    def clear_documents(self):
        """Clear all processed documents"""
        try:
            self.vector_store.clear()
            logger.info("All documents cleared from vector store")
        except Exception as e:
            logger.error(f"Error clearing documents: {str(e)}")
            raise Exception(f"Failed to clear documents: {str(e)}")
    
    def get_pipeline_stats(self) -> Dict:
        """Get statistics about the pipeline"""
        return {
            'embedding_model': self.embedding_model,
            'llm_model': self.llm_model,
            'ollama_url': self.ollama_url,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'document_count': self.get_document_count(),
            'vector_store_stats': self.vector_store.get_stats()
        }

async def test_rag_pipeline():
    """
    Test the complete RAG pipeline
    """
    try:
        logger.info("Testing RAG pipeline...")
        
        # Initialize pipeline
        pipeline = RAGPipeline()
        
        # Test query without documents
        try:
            answer, sources = await pipeline.query("test query")
            logger.info("Query without documents test passed")
        except:
            logger.info("Query without documents correctly raised exception")
        
        logger.info("RAG pipeline test completed!")
        return True
        
    except Exception as e:
        logger.error(f"RAG pipeline test failed: {str(e)}")
        return False

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_rag_pipeline())
