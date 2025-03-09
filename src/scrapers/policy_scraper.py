import os
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import logging
from src.config.config import Config
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolicyScraper:
    def __init__(self):
        self.config = Config()
        
    def fetch_document(self, url: str) -> str:
        """
        Fetch document content from a given URL
        
        Args:
            url (str): URL to fetch the document from
            
        Returns:
            str: Document content
        """
        try:
            logger.info(f"Attempting to fetch document from URL: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Log response status and content type
            logger.info(f"Response status: {response.status_code}")
            logger.info(f"Content type: {response.headers.get('content-type', 'unknown')}")
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the main content area (adjust selector based on website structure)
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
            if main_content:
                soup = main_content
                logger.info("Found main content area")
            else:
                logger.warning("Could not find main content area, using entire page")
            
            # Remove script and style elements
            for element in soup(["script", "style", "nav", "header", "footer"]):
                element.decompose()
                
            # Get text content
            text = soup.get_text()
            
            # Clean up text (breaking long line)
            lines = (line.strip() for line in text.splitlines())
            chunks = (
                phrase.strip() 
                for line in lines 
                for phrase in line.split("  ")
            )
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            logger.info(
                f"Successfully fetched document from {url} "
                f"({len(text)} characters)"
            )
            
            # Log a preview of the content
            preview = text[:200] + "..." if len(text) > 200 else text
            logger.info(f"Content preview: {preview}")
            
            return text
            
        except requests.RequestException as e:
            logger.error(
                f"Error fetching document from {url}. "
                f"Error type: {type(e).__name__}, Details: {str(e)}"
            )
            return ""
            
    def save_document(self, content: str, url: str) -> str:
        """
        Save document content to file
        
        Args:
            content (str): Document content to save
            url (str): Source URL of the document
            
        Returns:
            str: Path to saved file
        """
        if not content.strip():
            logger.warning(f"Empty content for URL {url}, skipping save")
            return ""
            
        filename = url.split('/')[-1].replace('.', '_') + '.txt'
        filepath = os.path.join(self.config.RAW_DOCS_DIR, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(
                f"Saved document to: {filepath} "
                f"(size: {len(content)} characters)"
            )
            return filepath
        except IOError as e:
            logger.error(f"Error saving document to {filepath}: {str(e)}")
            return ""
            
    def scrape_policies(self) -> List[Dict[str, str]]:
        """
        Scrape all policy documents from configured URLs
        
        Returns:
            List[Dict[str, str]]: List of dictionaries containing document info
        """
        documents = []
        total_urls = len(self.config.POLICY_URLS)
        logger.info(f"Starting to scrape {total_urls} URLs")
        
        for i, url in enumerate(
            tqdm(self.config.POLICY_URLS, desc="Scraping policy documents"),
            1
        ):
            logger.info(f"\nProcessing URL {i}/{total_urls}: {url}")
            
            if not url:
                logger.warning("Empty URL found, skipping")
                continue
                
            content = self.fetch_document(url)
            if not content:
                logger.warning(f"No content fetched from {url}, skipping")
                continue
                
            filepath = self.save_document(content, url)
            if filepath:
                doc_info = {
                    'url': url,
                    'filepath': filepath,
                    'content': content
                }
                documents.append(doc_info)
                logger.info(
                    f"Successfully processed document {i}: "
                    f"{url} (size: {len(content)} chars)"
                )
            
        logger.info(
            f"\nScraping complete. Successfully processed "
            f"{len(documents)}/{total_urls} documents"
        )
        return documents 