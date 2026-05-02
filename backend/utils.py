"""
Utility functions for PDF processing and text chunking
"""

import logging
from pathlib import Path
from typing import List, Tuple
import PyPDF2
import re

logger = logging.getLogger(__name__)

def validate_pdf_file(file_path: str) -> bool:
    """
    Validate if the file is a valid PDF
    """
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            # Check if PDF has pages
            if len(reader.pages) == 0:
                return False
            return True
    except Exception as e:
        logger.error(f"Error validating PDF file: {str(e)}")
        return False

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text content from PDF file
    """
    try:
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            # Extract text from each page
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text.strip():
                    text += page_text + "\n"
                logger.info(f"Extracted text from page {page_num + 1}")
        
        return text.strip()
        
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise Exception(f"Failed to extract text from PDF: {str(e)}")

def clean_text(text: str) -> str:
    """
    Clean and normalize text content
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\\]', '', text)
    
    # Clean up line breaks
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into chunks with specified size and overlap
    """
    if not text:
        return []
    
    # Clean the text first
    text = clean_text(text)
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Calculate end position for this chunk
        end = start + chunk_size
        
        # If this is the last chunk, take whatever is left
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # Try to break at a sentence boundary
        chunk = text[start:end]
        
        # Look for sentence endings near the chunk boundary
        sentence_endings = ['.', '!', '?', '\n']
        best_break = -1
        
        # Search backwards from the end for a good breaking point
        for i in range(len(chunk) - 1, max(0, len(chunk) - 100), -1):
            if chunk[i] in sentence_endings:
                best_break = i + 1
                break
        
        if best_break > 0:
            # Use the sentence boundary
            final_chunk = chunk[:best_break].strip()
            start += best_break
        else:
            # No good breaking point, use the chunk as is
            final_chunk = chunk.strip()
            start += chunk_size
        
        if final_chunk:
            chunks.append(final_chunk)
        
        # Calculate next start position with overlap
        if start < len(text):
            start = max(start - overlap, 0)
    
    logger.info(f"Created {len(chunks)} chunks from text")
    return chunks

def add_chunk_metadata(chunks: List[str], source_file: str) -> List[dict]:
    """
    Add metadata to each chunk
    """
    chunk_data = []
    
    for i, chunk in enumerate(chunks):
        chunk_info = {
            'id': f"{source_file}_chunk_{i}",
            'text': chunk,
            'source': source_file,
            'chunk_index': i,
            'metadata': {
                'source_file': source_file,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'char_count': len(chunk)
            }
        }
        chunk_data.append(chunk_info)
    
    return chunk_data

def process_pdf_document(file_path: str, chunk_size: int = 500, overlap: int = 100) -> List[dict]:
    """
    Complete pipeline to process a PDF document and return chunked data
    """
    try:
        # Validate PDF
        if not validate_pdf_file(file_path):
            raise ValueError("Invalid PDF file")
        
        # Extract text
        logger.info(f"Extracting text from: {file_path}")
        text = extract_text_from_pdf(file_path)
        
        if not text:
            raise ValueError("No text extracted from PDF")
        
        # Chunk the text
        logger.info("Chunking text...")
        chunks = chunk_text(text, chunk_size, overlap)
        
        if not chunks:
            raise ValueError("No chunks created from text")
        
        # Add metadata
        source_file = Path(file_path).name
        chunk_data = add_chunk_metadata(chunks, source_file)
        
        logger.info(f"Successfully processed {source_file}: {len(chunks)} chunks")
        return chunk_data
        
    except Exception as e:
        logger.error(f"Error processing PDF document: {str(e)}")
        raise Exception(f"Failed to process PDF document: {str(e)}")

def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in text (rough approximation)
    """
    # Rough estimation: ~4 characters per token for English text
    return len(text) // 4

def filter_chunks_by_length(chunks: List[dict], min_length: int = 50, max_length: int = 2000) -> List[dict]:
    """
    Filter chunks by length to remove very short or very long chunks
    """
    filtered_chunks = []
    
    for chunk in chunks:
        text_length = len(chunk['text'])
        if min_length <= text_length <= max_length:
            filtered_chunks.append(chunk)
    
    logger.info(f"Filtered chunks: {len(chunks)} -> {len(filtered_chunks)}")
    return filtered_chunks

# Utility function for debugging
def print_sample_chunks(chunks: List[dict], num_samples: int = 3):
    """
    Print sample chunks for debugging
    """
    print(f"\n=== Sample Chunks (showing {num_samples} of {len(chunks)}) ===")
    
    for i, chunk in enumerate(chunks[:num_samples]):
        print(f"\nChunk {i + 1}:")
        print(f"Source: {chunk['source']}")
        print(f"Length: {len(chunk['text'])} chars")
        print(f"Preview: {chunk['text'][:200]}...")
        print("-" * 50)
