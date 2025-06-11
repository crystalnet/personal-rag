# app.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, UnstructuredExcelLoader
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
import nltk
import ssl
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

# Load environment variables at the start
load_dotenv()

@dataclass
class EmbedderConfig:
    """Configuration for the document embedder."""
    chunk_size: int
    chunk_overlap: int
    max_retries: int
    retry_delay: int

    @classmethod
    def from_env(cls) -> 'EmbedderConfig':
        """Create configuration from environment variables."""
        return cls(
            chunk_size=int(os.getenv("CHUNK_SIZE")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP")),
            max_retries=int(os.getenv("MAX_RETRIES")),
            retry_delay=int(os.getenv("RETRY_DELAY"))
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

    def _load_documents(self, directory: str, glob_pattern: str) -> List[Any]:
        """Load documents from the specified directory."""
        print(f"📂 Loading documents from: {directory} with pattern: {glob_pattern}")
        loader = DirectoryLoader(directory, glob=glob_pattern, loader_cls=UnstructuredExcelLoader)
        docs = loader.load()
        
        if not docs:
            raise ValueError("No documents were loaded successfully!")
        
        print(f"✅ Successfully loaded {len(docs)} documents")
        self._print_document_stats(docs)
        return docs

    def _print_document_stats(self, docs: List[Any]) -> None:
        """Print statistics about loaded documents."""
        print("\n📊 Document types found:")
        doc_types = {}
        for doc in docs:
            file_type = os.path.splitext(doc.metadata.get('source', ''))[1]
            doc_types[file_type] = doc_types.get(file_type, 0) + 1
            print(f"\nDocument content preview: {doc.page_content[:200]}...")
        for file_type, count in doc_types.items():
            print(f"  - {file_type}: {count} files")

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
        
        # Generate deterministic IDs
        ids = [self._make_deterministic_id(doc, i) for i, doc in enumerate(chunked_docs)]
        print(f"Generated {len(ids)} unique IDs for chunks")

        # Create vector store
        vector_store = PineconeVectorStore(
            index=self.pc.Index(self.index_name),
            embedding=self.embeddings
        )

        # Get initial stats
        initial_stats = self.pc.Index(self.index_name).describe_index_stats()
        print(f"📊 Initial index stats: {initial_stats}")

        try:
            vector_store.add_documents(documents=chunked_docs, ids=ids)
            print("✅ Documents added to vector store")
        except Exception as e:
            raise Exception(f"Error adding documents to vector store: {str(e)}")

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

    def process_documents(self, directory: str, glob_pattern: str) -> None:
        """Process and upload documents from the specified directory."""
        try:
            docs = self._load_documents(directory, glob_pattern)
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
        glob_pattern=os.getenv("GLOB_PATTERN")
    )

if __name__ == "__main__":
    main()


