import re
from typing import List, Dict
import os
import logging
from src.config.config import Config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextProcessor:
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 100):
        """Initialize the text processor."""
        self.config = Config()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        # Remove multiple newlines and spaces
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Remove any non-breaking spaces and other special whitespace
        text = text.replace('\xa0', ' ')
        
        # Normalize quotes and dashes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace('–', '-').replace('—', '-')
        
        # Remove any remaining special characters
        text = re.sub(r'[^\w\s\.,;:!?\-\'\"()\[\]{}]', ' ', text)
        
        # Clean up final whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
        
    def split_into_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        # First clean the text
        text = self.clean_text(text)
        
        # Log the cleaned text length
        logger.info(f"Cleaned text length: {len(text)} characters")
        
        # Split into sentences (roughly)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if (
                current_length + sentence_length > self.chunk_size 
                and current_chunk
            ):
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Keep last sentences for overlap
                overlap_size = 0
                overlap_chunk = []
                
                for s in reversed(current_chunk):
                    if overlap_size + len(s) <= self.chunk_overlap:
                        overlap_chunk.insert(0, s)
                        overlap_size += len(s)
                    else:
                        break
                
                current_chunk = overlap_chunk
                current_length = overlap_size
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # Log chunk information
        logger.info(f"Split text into {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            logger.info(f"Chunk {i+1} length: {len(chunk)} characters")
            logger.debug(f"Chunk {i+1} preview: {chunk[:100]}...")
        
        return chunks
        
    def process_document(self, content: str, metadata: Dict) -> List[Dict]:
        """Process a document into chunks with metadata"""
        # Clean and split the text into chunks
        chunks = self.split_into_chunks(content)
        
        # Log document processing
        logger.info(f"Processing document from {metadata.get('source_url', 'unknown')}")
        logger.info(f"Document split into {len(chunks)} chunks")
        
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            # Create unique chunk ID
            chunk_id = f"{os.path.basename(metadata['source_file'])}_{i}"
            
            # Create chunk metadata
            chunk_metadata = {
                'chunk_id': chunk_id,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'content': chunk,
                'source_url': metadata.get('source_url', metadata.get('url', '')),
                'source_file': metadata.get('source_file', metadata.get('filepath', '')),
                'title': metadata.get('title', ''),
            }
            
            # Log chunk details
            logger.debug(f"Created chunk {chunk_id} with {len(chunk)} characters")
            logger.debug(f"Chunk preview: {chunk[:100]}...")
            
            processed_chunks.append(chunk_metadata)
        
        logger.info(f"Successfully processed {len(processed_chunks)} chunks")
        return processed_chunks
        
    def _is_likely_navigation(self, text: str) -> bool:
        """
        Check if text is likely navigation content
        
        Args:
            text (str): Text to check
            
        Returns:
            bool: True if text is likely navigation content
        """
        nav_indicators = [
            'menu', 'click here', 'home page', 'contact us',
            'privacy policy', 'terms of use', 'copyright',
            'all rights reserved', 'follow us', 'share this'
        ]
        
        # Count how many navigation indicators are present
        indicator_count = sum(
            1 for ind in nav_indicators if ind in text.lower()
        )
        
        # If more than 2 indicators are present, likely navigation
        return indicator_count > 2
        
    def save_processed_chunks(
        self, chunks: List[Dict[str, str]], document: Dict[str, str]
    ) -> str:
        """
        Save processed chunks to file
        
        Args:
            chunks (List[Dict[str, str]]): List of processed chunks
            document (Dict[str, str]): Original document metadata
            
        Returns:
            str: Path to saved processed file
        """
        filename = os.path.basename(document['filepath'])
        processed_path = os.path.join(
            self.config.PROCESSED_DOCS_DIR,
            f"processed_{filename}"
        )
        
        try:
            with open(processed_path, 'w', encoding='utf-8') as f:
                for chunk in chunks:
                    f.write(f"CHUNK {chunk['chunk_index']}\n")
                    f.write(f"SOURCE: {chunk['source_url']}\n")
                    f.write(chunk['content'])
                    f.write('\n\n---\n\n')
            return processed_path
        except IOError as e:
            logger.error(
                f"Error saving processed chunks to {processed_path}: {e}"
            )
            return "" 