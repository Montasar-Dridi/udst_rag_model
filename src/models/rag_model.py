from typing import List, Optional, Tuple
import logging
from src.config.config import Config
from src.scrapers.policy_scraper import PolicyScraper
from src.utils.text_processor import TextProcessor
from src.embeddings.embedder import DocumentEmbedder
from src.retrieval.faiss_index import FAISSIndex
from mistralai import Mistral


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGModel:
    def __init__(self):
        """Initialize the RAG model with all necessary components."""
        self.config = Config()
        self.scraper = PolicyScraper()
        self.processor = TextProcessor()
        self.embedder = DocumentEmbedder()
        self.index = FAISSIndex()
        self.client = Mistral(
            api_key=self.config.MISTRAL_API_KEY
        )
    
    def initialize(self) -> bool:
        """
        Initialize the system by loading existing index or creating new one
        
        Returns:
            bool: True if initialization was successful
        """
        # Try to load existing index
        if self.index.load_index():
            logger.info("Successfully loaded existing index")
            return True
            
        # If loading fails, create new index
        try:
            # Scrape documents
            documents = self.scraper.scrape_policies()
            if not documents:
                logger.error("No documents were scraped")
                return False
                
            # Process documents
            all_chunks = []
            for doc in documents:
                # Ensure consistent metadata keys
                metadata = {
                    'source_url': doc['url'],
                    'source_file': doc['filepath']
                }
                chunks = self.processor.process_document(
                    content=doc['content'],
                    metadata=metadata
                )
                all_chunks.extend(chunks)
                
            # Generate embeddings
            embeddings_dict = self.embedder.embed_chunks(all_chunks)
            
            # Create index
            self.index.create_index(embeddings_dict)
            logger.info("Successfully created new index")
            return True
            
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            return False
            
    def get_relevant_context(self, query: str) -> Tuple[str, List[str]]:
        """
        Retrieve relevant context and sources for a query
        
        Args:
            query (str): User query
            
        Returns:
            Tuple containing:
                - str: Concatenated context
                - List[str]: List of source URLs
        """
        logger.info(f"Processing query: {query}")
        
        # Generate query embedding
        query_embedding = self.embedder.generate_embedding(query)
        logger.info("Generated query embedding")
        
        # Search for relevant chunks
        results = self.index.search(query_embedding)
        logger.info(f"Search returned {len(results)} results")
        
        if not results:
            logger.warning("No relevant chunks found for query")
            return "", []
        
        # Sort results by similarity score
        results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        # Extract sources and log them
        sources = list(set(r['source_url'] for r in results))
        logger.info(f"Found {len(sources)} unique sources")
        for source in sources:
            logger.info(f"Source URL: {source}")
        
        # Build context string with chunks and their scores
        context_parts = []
        for r in results:
            score = r.get('similarity_score', 0)
            distance = r.get('distance', float('inf'))
            context_parts.append(
                f"Source: {r['source_url']}\n"
                f"Relevance Score: {score:.3f} (Distance: {distance:.3f})\n"
                f"Content: {r['content']}"
            )
        
        context = "\n\n---\n\n".join(context_parts)
        logger.info(f"Built context with {len(context_parts)} chunks")
        
        return context, sources
        
    def get_answer_with_sources(
        self, question: str
    ) -> Tuple[Optional[str], List[str]]:
        """
        Get answer and source documents for a question using RAG
        
        Args:
            question (str): User question
            
        Returns:
            Tuple containing:
                - Optional[str]: Generated answer
                - List[str]: List of source URLs
        """
        try:
            # Get relevant context and sources
            context, sources = self.get_relevant_context(question)
            if not context:
                return (
                    "I apologize, but I couldn't find any relevant information "
                    "in the available documents to answer your question.",
                    []  # Return empty sources list when no context found
                )
                
            # Create messages for Mistral chat
            system_msg = (
                "You are a helpful assistant that answers questions about "
                "UDST policies. Use the provided context to answer questions "
                "accurately. If you don't find relevant information in the "
                "context to answer the question, say so and don't include "
                "any source documents."
            )
            messages = [
                {
                    "role": "system",
                    "content": system_msg
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}"
                }
            ]
            
            # Generate answer using Mistral AI
            response = self.client.chat.complete(
                model=self.config.MISTRAL_MODEL,
                messages=messages,
                temperature=self.config.TEMPERATURE,
                max_tokens=self.config.MAX_TOKENS
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Only return sources if the answer indicates we found relevant information
            if any(phrase in answer.lower() for phrase in [
                "i apologize",
                "i'm sorry",
                "i am sorry",
                "no relevant information",
                "cannot find",
                "don't have information",
                "do not have information"
            ]):
                return answer, []
            
            return answer, sources
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return None, []
            
    def get_answer(self, question: str) -> Optional[str]:
        """
        Get answer for a question using RAG (legacy method)
        
        Args:
            question (str): User question
            
        Returns:
            Optional[str]: Generated answer or None if error occurs
        """
        answer, _ = self.get_answer_with_sources(question)
        return answer 