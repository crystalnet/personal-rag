# app.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader
)
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import nltk
import ssl
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

# Load environment variables at the start
load_dotenv()

class RobustTextLoader(TextLoader):
    """A text loader that handles different encodings."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"🔍 RobustTextLoader initialized for: {self.file_path}")

    def load(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            try:
                with open(self.file_path, "r", encoding="latin-1") as f:
                    text = f.read()
            except Exception as e:
                print(f"❌ Skipping {self.file_path}: cannot decode as UTF-8 or Latin-1 ({e})")
                return []

        return [Document(page_content=text, metadata={"source": str(self.file_path)})]

    def lazy_load(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            try:
                with open(self.file_path, "r", encoding="latin-1") as f:
                    text = f.read()
            except Exception as e:
                print(f"❌ Skipping {self.file_path}: cannot decode as UTF-8 or Latin-1 ({e})")
                return []

        return [Document(page_content=text, metadata={"source": str(self.file_path)})]

@dataclass
class EmbedderConfig:
    """Configuration for the document embedder."""
    chunk_size: int
    chunk_overlap: int
    max_retries: int
    retry_delay: int
    batch_size: int

    @classmethod
    def from_env(cls) -> 'EmbedderConfig':
        """Create configuration from environment variables."""
        return cls(
            chunk_size=int(os.getenv("CHUNK_SIZE")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP")),
            max_retries=int(os.getenv("MAX_RETRIES")),
            retry_delay=int(os.getenv("RETRY_DELAY")),
            batch_size=int(os.getenv("BATCH_SIZE"))
        )

class DocumentEmbedder:
    def __init__(self, config: EmbedderConfig = None):
        """Initialize the document embedder with configuration."""
        self.config = config or EmbedderConfig.from_env()
        self._setup_nltk()
        self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self._validate_pinecone_index()

    def _setup_nltk(self) -> None:
        """Set up NLTK punkt data."""
        try:
            nltk.data.find('tokenizers/punkt')
            print("✅ NLTK punkt data already available")
        except LookupError:
            print("📥 Downloading NLTK punkt data...")
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            nltk.download("punkt")

    def _validate_pinecone_index(self) -> None:
        """Validate that the Pinecone index exists."""
        indexes = self.pc.list_indexes()
        if self.index_name not in [ix.name for ix in indexes]:
            raise ValueError(f"Index '{self.index_name}' not found in Pinecone!")

    def _get_loader_for_pattern(self, pattern: str) -> Any:
        """Get the appropriate loader for the given file pattern."""
        if pattern.endswith('.xlsx'):
            return UnstructuredExcelLoader  # Using UnstructuredExcelLoader which uses openpyxl internally
        elif pattern.endswith('.docx'):
            return Docx2txtLoader  # Using Docx2txtLoader which uses docx2txt
        elif pattern.endswith('.pdf'):
            return PyPDFLoader  # Using PyPDFLoader which uses pypdf
        elif pattern.endswith('.txt'):
            return RobustTextLoader  # Using our robust text loader
        else:
            return None

    def _load_documents(self, directory: str, glob_patterns: str) -> List[Any]:
        """Load documents from the specified directory using multiple glob patterns."""
        print(f"Loading documents from {directory} with patterns: {glob_patterns}")

        # Check if directory exists
        if not os.path.exists(directory):
            raise ValueError(f"Directory does not exist: {directory}")
            
        # Parse glob patterns from environment variable
        patterns = [p.strip() for p in glob_patterns.split(',')]
            
        all_docs = []
        for pattern in patterns:
            print(f"\n🔍 Searching with pattern: {pattern}")
            matching_files = list(Path(directory).glob(pattern))
            print(f"📁 Found {len(matching_files)} files with pattern {pattern}:")
            
            if matching_files:
                print(f"\n📥 Loading documents...")
                loader_class = self._get_loader_for_pattern(pattern)
                if loader_class is None:
                    print(f"⚠️  No specific loader found for pattern {pattern}, using default loader")
                    loader = DirectoryLoader(directory, glob=pattern)
                else:
                    print(f"✅ Using {loader_class.__name__} for {pattern}")
                    loader = DirectoryLoader(directory, glob=pattern, loader_cls=loader_class)
                
                docs = loader.load()
                if docs:
                    all_docs.extend(docs)
                    print(f"✅ Successfully loaded {len(docs)} documents with pattern {pattern}")
        
        if not all_docs:
            raise ValueError(f"No documents were loaded successfully from directory '{directory}'")
        
        print(f"\n📊 Total documents loaded: {len(all_docs)}")
        return all_docs

    def _chunk_documents(self, docs: List[Any]) -> List[Any]:
        """Split documents into chunks."""
        print("\n✂️  Chunking documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        
        chunked_docs = text_splitter.split_documents(docs)
        print(f"✅ Split documents into {len(chunked_docs)} chunks")
        print(f"   Using chunk size: {self.config.chunk_size}, overlap: {self.config.chunk_overlap}")
        return chunked_docs

    def _make_deterministic_id(self, doc: Any, chunk_index: int) -> str:
        """Generate a deterministic ID for a document chunk."""
        content = doc.page_content
        source = doc.metadata.get("source", "")
        base = f"{Path(source).as_posix()}-{chunk_index}-{content.strip()}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()

    def _upload_to_pinecone(self, chunked_docs: List[Any]) -> None:
        """Upload documents to Pinecone with verification."""
        print("\n📤 Adding documents to vector store...")
        
        # Create vector store
        vector_store = PineconeVectorStore(
            index=self.pc.Index(self.index_name),
            embedding=self.embeddings
        )

        # Get initial stats
        initial_stats = self.pc.Index(self.index_name).describe_index_stats()
        print(f"📊 Initial index stats: {initial_stats}")

        # Process documents in batches
        total_chunks = len(chunked_docs)
        print(f"📦 Processing {total_chunks} chunks in batches of {self.config.batch_size}")
        
        for i in range(0, total_chunks, self.config.batch_size):
            batch = chunked_docs[i:i + self.config.batch_size]
            batch_ids = [self._make_deterministic_id(doc, i + j) for j, doc in enumerate(batch)]
            
            print(f"\n🔄 Processing batch {i//self.config.batch_size + 1}/{(total_chunks + self.config.batch_size - 1)//self.config.batch_size}")
            print(f"   Chunks {i+1}-{min(i+self.config.batch_size, total_chunks)} of {total_chunks}")
            
            try:
                vector_store.add_documents(documents=batch, ids=batch_ids)
                print(f"✅ Successfully uploaded batch")
            except Exception as e:
                print(f"❌ Error uploading batch: {str(e)}")
                raise

        self._verify_upload(initial_stats)

    def _verify_upload(self, initial_stats: Dict[str, Any]) -> None:
        """Verify that documents were successfully uploaded."""
        print("\n🔄 Verifying upload...")
        print(f"   Using max retries: {self.config.max_retries}, delay: {self.config.retry_delay}s")

        for attempt in range(self.config.max_retries):
            time.sleep(self.config.retry_delay)
            final_stats = self.pc.Index(self.index_name).describe_index_stats()
            print(f"📊 Stats check {attempt + 1}/{self.config.max_retries}: {final_stats}")
            
            if final_stats['total_vector_count'] > initial_stats['total_vector_count']:
                print("✅ Documents successfully uploaded to Pinecone!")
                return
            elif attempt < self.config.max_retries - 1:
                print(f"⏳ Waiting for index to update... (attempt {attempt + 1}/{self.config.max_retries})")
        
        print("❌ Warning: No new vectors were added to Pinecone after multiple checks. Please verify the upload manually.")

    def process_documents(self, directory: str, glob_patterns: str) -> None:
        """Process and upload documents from the specified directory."""
        try:
            docs = self._load_documents(directory, glob_patterns)
            chunked_docs = self._chunk_documents(docs)
            self._upload_to_pinecone(chunked_docs)
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            raise

def main():
    """Main entry point for the script."""
    # Create configuration from environment variables
    config = EmbedderConfig.from_env()

    # Create the embedder
    embedder = DocumentEmbedder(config)

    # Process documents with a more comprehensive glob pattern
    embedder.process_documents(
        directory=os.getenv("DOCS_DIRECTORY"),
        glob_patterns=os.getenv("GLOB_PATTERN")
    )

if __name__ == "__main__":
    main()


