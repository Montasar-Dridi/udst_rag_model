import pytest
import os
from src.models.rag_model import RAGModel
from src.config.config import Config
from mistralai.models.chat_completion import ChatMessage


@pytest.fixture
def rag_model():
    """Create a RAG model instance for testing."""
    return RAGModel()

@pytest.fixture
def config():
    """Create a config instance for testing."""
    return Config()

def test_model_initialization(rag_model):
    """Test that the model initializes correctly."""
    assert rag_model is not None
    assert rag_model.scraper is not None
    assert rag_model.processor is not None
    assert rag_model.embedder is not None
    assert rag_model.index is not None
    assert rag_model.client is not None

def test_get_answer(rag_model, monkeypatch):
    """Test the answer generation functionality."""
    # Mock the get_relevant_context method
    def mock_get_relevant_context(query):
        return "This is a mock context about UDST policies."
    
    def mock_chat(**kwargs):
        class MockResponse:
            class MockChoice:
                class MockMessage:
                    content = "This is a mock answer."
                message = MockMessage()
            choices = [MockChoice()]
        return MockResponse()
    
    monkeypatch.setattr(rag_model, 'get_relevant_context', 
                       mock_get_relevant_context)
    monkeypatch.setattr(rag_model.client, 'chat', mock_chat)
    
    # Test with a sample question
    question = "What is the attendance policy?"
    answer = rag_model.get_answer(question)
    
    assert answer is not None
    assert isinstance(answer, str)
    assert len(answer) > 0

def test_get_relevant_context(rag_model, monkeypatch):
    """Test the context retrieval functionality."""
    # Mock the necessary components
    def mock_generate_embedding(text):
        return [0.1] * rag_model.config.EMBEDDING_DIMENSION
        
    def mock_search(query_embedding):
        return [
            {
                'source_url': 'http://example.com',
                'content': 'Sample policy content'
            }
        ]
    
    monkeypatch.setattr(rag_model.embedder, 'generate_embedding',
                       mock_generate_embedding)
    monkeypatch.setattr(rag_model.index, 'search', mock_search)
    
    # Test context retrieval
    context = rag_model.get_relevant_context("test query")
    
    assert context is not None
    assert isinstance(context, str)
    assert "Sample policy content" in context

def test_initialization_with_no_documents(rag_model, monkeypatch):
    """Test initialization behavior when no documents are available."""
    def mock_scrape_policies():
        return []
    
    monkeypatch.setattr(rag_model.scraper, 'scrape_policies',
                       mock_scrape_policies)
    
    success = rag_model.initialize()
    assert not success

def test_initialization_with_documents(rag_model, monkeypatch):
    """Test initialization with mock documents."""
    def mock_scrape_policies():
        return [{
            'url': 'http://example.com',
            'filepath': 'test.txt',
            'content': 'Test content'
        }]
    
    monkeypatch.setattr(rag_model.scraper, 'scrape_policies',
                       mock_scrape_policies)
    monkeypatch.setattr(rag_model.index, 'create_index',
                       lambda x: None)
    
    success = rag_model.initialize()
    assert success 