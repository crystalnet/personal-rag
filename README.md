# Personal RAG System

A personal RAG (Retrieval-Augmented Generation) system that allows you to chat with your documents.

## System Dependencies

Before running the application, make sure you have the following system dependencies installed:

### macOS
```bash
brew install poppler  # Required for PDF processing
```

### Ubuntu/Debian
```bash
sudo apt-get install poppler-utils  # Required for PDF processing
```

### Windows
Download and install poppler from: https://github.com/oschwartz10612/poppler-windows/releases/
Make sure to add the poppler `bin` directory to your system PATH.

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables in `.env`:
```env
# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key

# Embedder Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_RETRIES=3
RETRY_DELAY=2

# Document Processing
DOCS_DIRECTORY=docs
GLOB_PATTERN=**/*.pdf,**/*.xlsx,**/*.docx,**/*.txt

# Chat Configuration
CHAT_STREAMING=true
CHAT_MODE=context
CHAT_MODEL=gpt-4-turbo-preview
CHAT_TEMPERATURE=0.7
```

## Usage

1. Place your documents in the `docs` directory
2. Run the embedder to process your documents:
```bash
python embedder.py
```

3. Start the chat interface:
```bash
python chat.py
```

## Supported File Types

- PDF files (requires poppler)
- Excel files (.xlsx)
- Word documents (.docx)
- Text files (.txt)

## Troubleshooting

If you encounter issues with PDF processing, make sure:
1. Poppler is installed correctly
2. The poppler binaries are in your system PATH
3. You have read permissions for the PDF files