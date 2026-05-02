"""
Vector store implementation using FAISS for efficient similarity search
"""

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import faiss

logger = logging.getLogger(__name__)

class FAISSVectorStore:
    """
    FAISS-based vector store for efficient similarity search
    """
    
    def __init__(self, embedding_dimension: int, index_type: str = "flat"):
        """
        Initialize the vector store
        
        Args:
            embedding_dimension: Dimension of embeddings
            index_type: Type of FAISS index ("flat", "ivf", "hnsw")
        """
        self.embedding_dimension = embedding_dimension
        self.index_type = index_type
        self.index = None
        self.documents = []  # Store document metadata
        self.embeddings = []  # Store raw embeddings
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index based on type"""
        try:
            if self.index_type == "flat":
                # Simple flat index (exact search)
                self.index = faiss.IndexFlatIP(self.embedding_dimension)  # Inner Product
                logger.info("Initialized FAISS Flat index")
            
            elif self.index_type == "ivf":
                # IVF index (approximate search)
                nlist = 100  # Number of clusters
                quantizer = faiss.IndexFlatIP(self.embedding_dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dimension, nlist)
                logger.info("Initialized FAISS IVF index")
            
            elif self.index_type == "hnsw":
                # HNSW index (hierarchical navigable small world)
                self.index = faiss.IndexHNSWFlat(self.embedding_dimension, 32)  # M=32
                logger.info("Initialized FAISS HNSW index")
            
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
                
        except Exception as e:
            logger.error(f"Error initializing FAISS index: {str(e)}")
            raise Exception(f"Failed to initialize index: {str(e)}")
    
    def add_documents(self, embeddings: np.ndarray, documents: List[Dict]):
        """
        Add documents and their embeddings to the vector store
        
        Args:
            embeddings: numpy array of document embeddings
            documents: list of document metadata
        """
        try:
            if len(embeddings) != len(documents):
                raise ValueError("Number of embeddings and documents must match")
            
            if embeddings.shape[1] != self.embedding_dimension:
                raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dimension}, got {embeddings.shape[1]}")
            
            # Ensure embeddings are in the right format
            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype(np.float32)
            
            # Normalize embeddings for better similarity search
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
            
            # Add to FAISS index
            self.index.add(embeddings)
            
            # Store documents and embeddings
            self.documents.extend(documents)
            self.embeddings.extend(embeddings.tolist())
            
            logger.info(f"Added {len(documents)} documents to vector store. Total: {len(self.documents)}")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise Exception(f"Failed to add documents: {str(e)}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[Dict], np.ndarray]:
        """
        Search for most similar documents
        
        Args:
            query_embedding: embedding of the query
            k: number of documents to retrieve
            
        Returns:
            Tuple of (documents, similarity_scores)
        """
        try:
            if len(self.documents) == 0:
                return [], np.array([])
            
            # Ensure query embedding is in right format
            if query_embedding.dtype != np.float32:
                query_embedding = query_embedding.astype(np.float32)
            
            # Normalize query embedding
            query_norm = np.linalg.norm(query_embedding)
            if query_norm > 0:
                query_embedding = query_embedding / query_norm
            
            # Reshape for FAISS (expects 2D array)
            query_embedding = query_embedding.reshape(1, -1)
            
            # Adjust k if necessary
            k = min(k, len(self.documents))
            
            # Search in FAISS index
            scores, indices = self.index.search(query_embedding, k)
            
            # Get documents and scores
            retrieved_docs = []
            retrieved_scores = scores[0]  # First (and only) query
            
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(self.documents):  # Valid index
                    doc = self.documents[idx].copy()
                    doc['similarity_score'] = float(retrieved_scores[i])
                    retrieved_docs.append(doc)
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents")
            return retrieved_docs, retrieved_scores
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            raise Exception(f"Failed to search documents: {str(e)}")
    
    def get_document_count(self) -> int:
        """Get the number of documents in the store"""
        return len(self.documents)
    
    def clear(self):
        """Clear all documents from the vector store"""
        try:
            self._initialize_index()  # Reinitialize index
            self.documents = []
            self.embeddings = []
            logger.info("Vector store cleared")
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            raise Exception(f"Failed to clear vector store: {str(e)}")
    
    def save_index(self, file_path: str):
        """
        Save the vector store to disk
        
        Args:
            file_path: path to save the index (without extension)
        """
        try:
            file_path = Path(file_path)
            file_path.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            index_path = file_path / "faiss.index"
            faiss.write_index(self.index, str(index_path))
            
            # Save documents and metadata
            metadata = {
                'documents': self.documents,
                'embeddings': self.embeddings,
                'embedding_dimension': self.embedding_dimension,
                'index_type': self.index_type
            }
            
            metadata_path = file_path / "metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Vector store saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise Exception(f"Failed to save vector store: {str(e)}")
    
    def load_index(self, file_path: str):
        """
        Load the vector store from disk
        
        Args:
            file_path: path to load the index from (without extension)
        """
        try:
            file_path = Path(file_path)
            
            # Load FAISS index
            index_path = file_path / "faiss.index"
            if not index_path.exists():
                raise FileNotFoundError(f"Index file not found: {index_path}")
            
            self.index = faiss.read_index(str(index_path))
            
            # Load documents and metadata
            metadata_path = file_path / "metadata.pkl"
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            self.documents = metadata['documents']
            self.embeddings = metadata['embeddings']
            self.embedding_dimension = metadata['embedding_dimension']
            self.index_type = metadata['index_type']
            
            logger.info(f"Vector store loaded from {file_path}. Documents: {len(self.documents)}")
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise Exception(f"Failed to load vector store: {str(e)}")
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        return {
            'document_count': len(self.documents),
            'embedding_dimension': self.embedding_dimension,
            'index_type': self.index_type,
            'index_ntotal': self.index.ntotal if self.index else 0,
            'is_trained': self.index.is_trained if self.index else False
        }

# Global vector store instance
_vector_store = None

def get_vector_store(embedding_dimension: int, index_type: str = "flat") -> FAISSVectorStore:
    """
    Get or create the global vector store instance
    
    Args:
        embedding_dimension: Dimension of embeddings
        index_type: Type of FAISS index
        
    Returns:
        FAISSVectorStore instance
    """
    global _vector_store
    
    if _vector_store is None:
        _vector_store = FAISSVectorStore(embedding_dimension, index_type)
    
    return _vector_store

def test_vector_store():
    """
    Test the vector store functionality
    """
    try:
        logger.info("Testing vector store functionality...")
        
        # Create test data
        embedding_dim = 384  # all-MiniLM-L6-v2 dimension
        store = FAISSVectorStore(embedding_dim)
        
        # Create test embeddings and documents
        test_embeddings = np.random.rand(5, embedding_dim).astype(np.float32)
        test_documents = [
            {'id': 'doc1', 'text': 'Test document 1', 'source': 'test1.pdf'},
            {'id': 'doc2', 'text': 'Test document 2', 'source': 'test1.pdf'},
            {'id': 'doc3', 'text': 'Test document 3', 'source': 'test2.pdf'},
            {'id': 'doc4', 'text': 'Test document 4', 'source': 'test2.pdf'},
            {'id': 'doc5', 'text': 'Test document 5', 'source': 'test3.pdf'}
        ]
        
        # Add documents
        store.add_documents(test_embeddings, test_documents)
        
        # Test search
        query = test_embeddings[0]  # Use first document as query
        results, scores = store.search(query, k=3)
        
        logger.info(f"Search results: {len(results)} documents")
        for i, (doc, score) in enumerate(zip(results, scores)):
            logger.info(f"  {i+1}. Score: {score:.4f} - {doc['id']}")
        
        # Test stats
        stats = store.get_stats()
        logger.info(f"Vector store stats: {stats}")
        
        logger.info("Vector store test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Vector store test failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Run test if this file is executed directly
    test_vector_store()
