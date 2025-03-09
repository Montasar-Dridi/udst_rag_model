import os
from dotenv import load_dotenv
from src.models.rag_model import RAGModel
import logging


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def main():
    # Initialize the RAG model
    logger.info("Initializing RAG model...")
    rag = RAGModel()
    
    # Initialize the system (this will either load or create the index)
    if not rag.initialize():
        logger.error("Failed to initialize RAG model")
        return
        
    # Example questions
    questions = [
        "What is the attendance policy?",
        "What are the requirements for academic probation?",
        "How can I apply for a leave of absence?"
    ]
    
    # Answer each question
    for question in questions:
        logger.info(f"\nQuestion: {question}")
        
        answer = rag.get_answer(question)
        if answer:
            logger.info(f"Answer: {answer}")
        else:
            logger.error("Failed to generate answer")

if __name__ == "__main__":
    main() 