from typing import List, Dict
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
from src.config.config import Config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentEmbedder:
    def __init__(self):
        """Initialize the document embedder with the specified model."""
        self.config = Config()
        self.model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text chunk
        
        Args:
            text (str): Text to embed
            
        Returns:
            np.ndarray: Embedding vector
        """
        try:
            with torch.no_grad():
                embedding = self.model.encode(text)
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return np.zeros(self.config.EMBEDDING_DIMENSION)
            
    def embed_chunks(self, chunks: List[Dict[str, str]]) -> Dict[str, Dict]:
        """
        Generate embeddings for multiple text chunks
        
        Args:
            chunks (List[Dict[str, str]]): List of text chunks with metadata
            
        Returns:
            Dict[str, Dict]: Dictionary mapping chunk IDs to embeddings and metadata
        """
        embeddings_dict = {}
        
        for chunk in chunks:
            chunk_id = chunk['chunk_id']
            embedding = self.generate_embedding(chunk['content'])
            
            # Copy all metadata except chunk_id which becomes the dict key
            metadata = chunk.copy()
            del metadata['chunk_id']
            
            embeddings_dict[chunk_id] = {
                'embedding': embedding,
                'metadata': metadata
            }
            
        return embeddings_dict 