# app.py
import os
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader,
    StorageContext, ServiceContext
)
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import PDFReader
from pathlib import Path

load_dotenv()

# 1️⃣  Load + chunk docs
try:
    # Specify file extensions we want to process
    file_extensions = {
        '.pdf': 'PDFReader',
        '.docx': 'DocxReader',
        '.xlsx': 'ExcelReader',
        '.txt': 'TextReader'
    }
    
    # Check for required dependencies
    required_packages = {
        '.docx': 'docx2txt',
        '.xlsx': 'openpyxl',
        '.pdf': 'pypdf'
    }
    
    # Verify dependencies are installed
    for ext, package in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            print(f"Error: {package} is required to read {ext} files. Please install it with 'pip install {package}'")
            exit(1)
    
    # Custom file extractor for PDFs to handle errors better
    def custom_pdf_reader(file_path: str):
        try:
            return PDFReader().load_data(file_path)
        except Exception as e:
            print(f"Warning: Error reading PDF {file_path}: {str(e)}")
            # Return empty list instead of raising error
            return []

    # Create custom file extractors
    file_extractors = {
        '.pdf': custom_pdf_reader
    }
    
    # Load documents with specific file types
    docs = SimpleDirectoryReader(
        input_dir="docs",
        recursive=True,
        required_exts=list(file_extensions.keys()),
        file_extractor=file_extractors,
        raise_on_error=False  # Don't raise errors on individual files
    ).load_data()
    
    if not docs:
        print("❌ No documents were loaded successfully!")
        exit(1)
    
    print(f"✅ Successfully loaded {len(docs)} documents")

except Exception as e:
    print(f"Error loading documents: {str(e)}")
    exit(1)

# 2️⃣  Connect to Pinecone
vector_store = PineconeVectorStore(
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    environment=os.getenv("PINECONE_ENV"),
    api_key=os.getenv("PINECONE_API_KEY"),
)

# 3️⃣  Create index (embeds + uploads)
service_context = ServiceContext.from_defaults(
    embed_model=OpenAIEmbedding(model="text-embedding-3-large")
)
index = VectorStoreIndex.from_documents(
    docs,
    vector_store=vector_store,
    service_context=service_context,
)

print("✅ Index built!")
