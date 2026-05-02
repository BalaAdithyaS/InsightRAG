"""
Embedding generation using Hugging Face sentence-transformers
"""

import logging
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Handles embedding generation using sentence-transformers
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model
        
        Args:
            model_name: Name of the sentence-transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence-transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            
            # Check if CUDA is available
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {device}")
            
            # Load the model
            self.model = SentenceTransformer(self.model_name, device=device)
            
            # Get embedding dimension
            test_embedding = self.model.encode("test", show_progress_bar=False)
            self.embedding_dim = test_embedding.shape[0]
            
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise Exception(f"Failed to load embedding model: {str(e)}")
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """
        Generate embeddings for given texts
        
        Args:
            texts: Single text or list of texts to encode
            batch_size: Batch size for processing multiple texts
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of embeddings
        """
        try:
            if self.model is None:
                raise ValueError("Model not loaded")
            
            # Convert single text to list
            if isinstance(texts, str):
                texts = [texts]
            
            if not texts:
                return np.array([])
            
            logger.info(f"Encoding {len(texts)} texts...")
            
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for better similarity search
            )
            
            logger.info(f"Successfully generated embeddings with shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise Exception(f"Failed to generate embeddings: {str(e)}")
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to encode
            
        Returns:
            numpy array of embedding
        """
        return self.encode(text)[0]
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings
        
        Returns:
            Embedding dimension
        """
        return self.embedding_dim
    
    def compute_similarity(self, query_embedding: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and document embeddings
        
        Args:
            query_embedding: Embedding of the query
            document_embeddings: Embeddings of documents
            
        Returns:
            Array of similarity scores
        """
        try:
            # Ensure embeddings are normalized
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            doc_norms = document_embeddings / np.linalg.norm(document_embeddings, axis=1, keepdims=True)
            
            # Compute cosine similarity
            similarities = np.dot(doc_norms, query_norm)
            
            return similarities
            
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            raise Exception(f"Failed to compute similarity: {str(e)}")
    
    def find_most_similar(self, query_embedding: np.ndarray, document_embeddings: np.ndarray, 
                         k: int = 5) -> tuple:
        """
        Find top-k most similar documents to query
        
        Args:
            query_embedding: Embedding of the query
            document_embeddings: Embeddings of documents
            k: Number of top documents to return
            
        Returns:
            Tuple of (indices, similarity_scores)
        """
        try:
            # Compute similarities
            similarities = self.compute_similarity(query_embedding, document_embeddings)
            
            # Get top-k indices and scores
            if len(similarities) <= k:
                top_indices = np.argsort(similarities)[::-1]
                top_scores = similarities[top_indices]
            else:
                top_indices = np.argpartition(similarities, -k)[-k:]
                top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
                top_scores = similarities[top_indices]
            
            return top_indices, top_scores
            
        except Exception as e:
            logger.error(f"Error finding most similar documents: {str(e)}")
            raise Exception(f"Failed to find similar documents: {str(e)}")

# Global embedding generator instance
_embedding_generator = None

def get_embedding_generator(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingGenerator:
    """
    Get or create the global embedding generator instance
    
    Args:
        model_name: Name of the model to use
        
    Returns:
        EmbeddingGenerator instance
    """
    global _embedding_generator
    
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator(model_name)
    
    return _embedding_generator

def create_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Convenience function to create embeddings
    
    Args:
        texts: List of texts to encode
        model_name: Name of the model to use
        
    Returns:
        numpy array of embeddings
    """
    generator = get_embedding_generator(model_name)
    return generator.encode(texts)

def test_embedding_functionality():
    """
    Test the embedding functionality with sample data
    """
    try:
        logger.info("Testing embedding functionality...")
        
        # Create embedding generator
        generator = get_embedding_generator()
        
        # Test single text
        test_text = "This is a test sentence for embedding generation."
        embedding = generator.encode_single(test_text)
        
        logger.info(f"Single text embedding shape: {embedding.shape}")
        logger.info(f"Embedding dimension: {generator.get_embedding_dimension()}")
        
        # Test multiple texts
        test_texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing helps computers understand human language."
        ]
        
        embeddings = generator.encode(test_texts)
        logger.info(f"Multiple texts embedding shape: {embeddings.shape}")
        
        # Test similarity
        query = "AI and machine learning"
        query_embedding = generator.encode_single(query)
        
        indices, scores = generator.find_most_similar(query_embedding, embeddings, k=2)
        
        logger.info("Top similar texts:")
        for i, (idx, score) in enumerate(zip(indices, scores)):
            logger.info(f"  {i+1}. Score: {score:.4f} - {test_texts[idx][:50]}...")
        
        logger.info("Embedding functionality test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Embedding test failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Run test if this file is executed directly
    test_embedding_functionality()
