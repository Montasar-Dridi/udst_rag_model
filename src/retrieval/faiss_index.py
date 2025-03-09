import faiss
import numpy as np
from typing import List, Dict
import json
import logging
from src.config.config import Config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FAISSIndex:
    def __init__(self):
        """Initialize FAISS index manager."""
        self.config = Config()
        self.index = None
        self.metadata = {}
        self.id_mapping = {}  # Map FAISS indices to chunk IDs
        
    def create_index(self, embeddings_dict: Dict[str, Dict]):
        """Create FAISS index from embeddings.
        
        Args:
            embeddings_dict: Dictionary of embeddings and metadata
        """
        chunk_ids = list(embeddings_dict.keys())
        embeddings = np.array([
            embeddings_dict[chunk_id]['embedding'] 
            for chunk_id in chunk_ids
        ])
        
        # Create and populate index
        self.index = faiss.IndexFlatL2(self.config.EMBEDDING_DIMENSION)
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata and mapping
        self.metadata = {}
        for i, chunk_id in enumerate(chunk_ids):
            self.metadata[chunk_id] = embeddings_dict[chunk_id]['metadata']
            self.id_mapping[i] = chunk_id
        
        # Save index and metadata
        self.save_index()
        
    def save_index(self):
        """Save FAISS index and metadata to disk."""
        if self.index is None:
            logger.error("No index to save")
            return
            
        try:
            # Save FAISS index
            faiss.write_index(self.index, self.config.INDEX_PATH)
            
            # Save metadata and mapping
            metadata_path = f"{self.config.INDEX_PATH}_metadata.json"
            save_data = {
                'metadata': self.metadata,
                'id_mapping': self.id_mapping
            }
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
                
            logger.info("Successfully saved index and metadata")
            
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            
    def load_index(self) -> bool:
        """
        Load FAISS index and metadata from disk
        
        Returns:
            bool: True if loading was successful, False otherwise
        """
        try:
            # Load FAISS index
            self.index = faiss.read_index(self.config.INDEX_PATH)
            
            # Load metadata and mapping
            metadata_path = f"{self.config.INDEX_PATH}_metadata.json"
            with open(metadata_path, 'r', encoding='utf-8') as f:
                save_data = json.load(f)
                self.metadata = save_data['metadata']
                self.id_mapping = {int(k): v for k, v in save_data['id_mapping'].items()}
                
            logger.info("Successfully loaded index and metadata")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return False
            
    def search(self, query_embedding: np.ndarray) -> List[Dict[str, str]]:
        """
        Search for similar documents using query embedding
        
        Args:
            query_embedding (np.ndarray): Query embedding vector
            
        Returns:
            List[Dict[str, str]]: List of similar documents with metadata
        """
        if self.index is None:
            logger.error("No index available for search")
            return []
            
        # Reshape query if needed
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        # Search index
        distances, indices = self.index.search(
            query_embedding.astype('float32'),
            self.config.TOP_K_MATCHES
        )
        
        # Log search results for debugging
        logger.info(f"Found {len(indices[0])} matches")
        logger.info(f"Distances: {distances[0]}")
        logger.info(f"Indices: {indices[0]}")
        
        # Get metadata for results
        results = []
        for i, idx in enumerate(indices[0]):
            distance = distances[0][i]
            
            # Get original chunk ID from mapping
            chunk_id = self.id_mapping.get(int(idx))
            if chunk_id is None:
                logger.warning(f"No mapping found for index {idx}")
                continue
                
            # Get metadata for chunk
            if chunk_id in self.metadata:
                result = self.metadata[chunk_id].copy()
                result['similarity_score'] = float(1.0 / (1.0 + distance))
                result['distance'] = float(distance)
                results.append(result)
                logger.info(
                    f"Added result from {result.get('source_url', 'unknown')}"
                )
            else:
                logger.warning(f"No metadata found for chunk_id {chunk_id}")
        
        logger.info(f"Returning {len(results)} results after filtering")
        if not results:
            logger.warning("No results found after filtering!")
            
        return results 