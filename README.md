# UDST Policy RAG System

A production-ready Retrieval-Augmented Generation (RAG) system for answering questions about UDST policies. This system scrapes policy documents, processes them, and uses advanced language models to provide accurate answers based on the policy content.

## Features

- Web scraping of UDST policy documents
- Text preprocessing and chunking
- Document embedding using state-of-the-art language models
- FAISS-based vector similarity search
- Question-answering using RAG architecture

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/udst-rag-model.git
cd udst-rag-model
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory and add your configuration:
```
OPENAI_API_KEY=your_api_key_here
POLICY_URLS=url1,url2,url3
```

## Project Structure

```
udst_rag_model/
├── src/
│   ├── config/         # Configuration files
│   ├── data/           # Data storage
│   ├── embeddings/     # Embedding generation
│   ├── models/         # Model definitions
│   ├── retrieval/      # FAISS retrieval system
│   ├── scrapers/       # Web scraping utilities
│   └── utils/          # Helper functions
├── tests/              # Unit tests
├── docs/               # Documentation
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Usage

1. Initialize the system:
```python
from src.models.rag_model import RAGModel

rag = RAGModel()
rag.initialize()
```

2. Ask questions:
```python
question = "What is the attendance policy?"
answer = rag.get_answer(question)
print(answer)
```

## Development

- Run tests: `pytest tests/`
- Format code: `black .`
- Check linting: `flake8`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License 