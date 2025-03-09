import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class Config:
    # API Keys
    MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
    
    # URLs for policy documents
    _raw_urls = os.getenv('POLICY_URLS', '')
    # Split on commas and clean up each URL
    _url_list = [
        url.strip()
        for url in _raw_urls.split(',')
        if url.strip() and url.strip().startswith('http')
    ]
    
    # Remove duplicates while preserving order
    POLICY_URLS = list(dict.fromkeys(_url_list))
    
    # Log URLs for debugging
    logger.info(f"Found {len(POLICY_URLS)} URLs to process:")
    for url in POLICY_URLS:
        logger.info(f"URL: {url}")
    
    # Embedding Configuration
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_DIMENSION = 768
    CHUNK_SIZE = 300
    CHUNK_OVERLAP = 100
    
    # FAISS Configuration
    INDEX_PATH = "src/data/faiss_index"
    
    # Storage Configuration
    DATA_DIR = "src/data"
    RAW_DOCS_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DOCS_DIR = os.path.join(DATA_DIR, "processed")
    
    # Create directories if they don't exist
    os.makedirs(RAW_DOCS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DOCS_DIR, exist_ok=True)
    
    # Retrieval Configuration
    TOP_K_MATCHES = 10
    SIMILARITY_THRESHOLD = 1.0
    
    # Model Configuration
    TEMPERATURE = 0.7
    MAX_TOKENS = 500
    MISTRAL_MODEL = "mistral-medium"
    
    # Additional configuration
    # ... (keep the existing attributes)
    
    # ... (keep the existing methods)
    
    # ... (keep the existing initialization) 