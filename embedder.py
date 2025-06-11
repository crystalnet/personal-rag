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
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from queue import Queue
from threading import Thread, Event
import traceback
import logging
from datetime import datetime

# Load environment variables at the start
load_dotenv()

# Set up logging
def setup_logging():
    """Configure logging to write to both file and console."""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"embedder_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This will still show logs in terminal
        ]
    )
    
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging()

class RobustTextLoader(TextLoader):
    """A text loader that handles different encodings."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info(f"RobustTextLoader initialized for: {self.file_path}")

    def load(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            try:
                with open(self.file_path, "r", encoding="latin-1") as f:
                    text = f.read()
            except Exception as e:
                logger.error(f"Skipping {self.file_path}: cannot decode as UTF-8 or Latin-1 ({e})")
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
        
        # Initialize queues
        self.read_queue = Queue()
        self.chunk_queue = Queue()
        self.embed_queue = Queue()
        
        # Initialize error tracking
        self.error_event = Event()
        self.errors = []
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        
        # Initialize progress tracking
        self.total_files = 0
        self.files_read = 0
        self.files_chunked = 0
        self.total_chunks = 0
        self.chunks_embedded = 0
        self.chunks_uploaded = 0

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

    def _handle_error(self, error: Exception, stage: str, context: str = ""):
        """Handle errors in the pipeline."""
        error_msg = f"Error in {stage}: {str(error)}\nContext: {context}\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        self.errors.append(error_msg)
        self.error_event.set()

    def _print_progress(self):
        """Print current progress across all stages."""
        # Clear previous lines and move to start
        print("\033[2K\033[1A" * 3, end="")  # Clear 3 lines and move up
        print("\033[K", end="")  # Clear current line
        
        # Print progress information
        print(f"📊 Progress:")
        print(f"  Files: {self.files_read}/{self.total_files} read, "
              f"{self.files_chunked}/{self.total_files} chunked")
        if self.total_chunks > 0:
            print(f"  Chunks: {self.chunks_embedded}/{self.total_chunks} embedded "
                  f"({(self.chunks_embedded/self.total_chunks)*100:.1f}%), "
                  f"{self.chunks_uploaded}/{self.total_chunks} uploaded "
                  f"({(self.chunks_uploaded/self.total_chunks)*100:.1f}%)")
        
        # Flush to ensure immediate display
        import sys
        sys.stdout.flush()

    def _reader_worker(self, file_paths: List[Path]):
        """Read documents and add them to the chunking queue."""
        try:
            self.total_files = len(file_paths)
            
            for file_path in file_paths:
                try:
                    # Get the appropriate loader for this file
                    loader_class = self._get_loader_for_pattern(file_path.name)
                    if not loader_class:
                        print(f"\n⚠️  No loader found for {file_path}")
                        continue
                    
                    # Load the document
                    loader = loader_class(str(file_path))
                    docs = loader.load()
                    
                    # Add to chunking queue
                    self.read_queue.put((file_path, docs))
                    self.files_read += 1
                    self._print_progress()
                    
                except Exception as e:
                    error_msg = f"Error reading {file_path}: {str(e)}"
                    print(f"\n❌ {error_msg}")
                    self.errors.append(error_msg)
                    self.error_event.set()
            
            # Signal end of reading
            self.read_queue.put(None)
            
        except Exception as e:
            print(f"\n❌ Fatal error in reader: {str(e)}")
            self.error_event.set()
            self.read_queue.put(None)  # Signal end even if there's an error

    def _chunker_worker(self):
        """Worker for chunking documents."""
        try:
            while True:
                if self.error_event.is_set():
                    break
                    
                item = self.read_queue.get()
                if item is None:  # End signal
                    self.read_queue.task_done()
                    break
                    
                file_path, docs = item
                try:
                    chunks = self.text_splitter.split_documents(docs)
                    self.total_chunks += len(chunks)
                    self.chunk_queue.put((file_path, chunks))
                    self.files_chunked += 1
                    self._print_progress()
                except Exception as e:
                    self._handle_error(e, "chunker", f"Failed to chunk {file_path}")
                finally:
                    self.read_queue.task_done()
        finally:
            self.chunk_queue.put(None)  # Signal end of chunking

    def _embedder_worker(self):
        """Worker for embedding documents."""
        try:
            logger.info("Starting embedder worker...")
            batch_count = 0
            while True:
                if self.error_event.is_set():
                    logger.error("Embedder worker stopping due to error event")
                    break
                    
                item = self.chunk_queue.get()
                if item is None:  # End signal
                    logger.info("Embedder worker received end signal")
                    self.chunk_queue.task_done()
                    break
                    
                file_path, chunks = item
                try:
                    # Process chunks in batches
                    for i in range(0, len(chunks), self.config.batch_size):
                        batch = chunks[i:i + self.config.batch_size]
                        batch_count += 1
                        logger.info(f"Embedding batch {batch_count} from {file_path} ({len(batch)} chunks)")
                        vectors = self.embeddings.embed_documents([doc.page_content for doc in batch])
                        logger.info(f"Batch {batch_count} embedded successfully")
                        self.embed_queue.put((file_path, batch, vectors))
                        self.chunks_embedded += len(batch)
                        self._print_progress()
                except Exception as e:
                    logger.error(f"Error embedding batch from {file_path}: {str(e)}")
                    self._handle_error(e, "embedder", f"Failed to embed {file_path}")
                finally:
                    self.chunk_queue.task_done()
        except Exception as e:
            logger.error(f"Fatal error in embedder worker: {str(e)}")
            self._handle_error(e, "embedder", "Fatal error in embedder worker")
        finally:
            logger.info("Embedder worker finished")
            self.embed_queue.put(None)  # Signal end of embedding

    def _uploader_worker(self):
        """Worker for uploading to Pinecone."""
        try:
            logger.info("Starting uploader worker...")
            vector_store = PineconeVectorStore(
                index=self.pc.Index(self.index_name),
                embedding=self.embeddings
            )
            
            batch_count = 0
            while True:
                if self.error_event.is_set():
                    logger.error("Uploader worker stopping due to error event")
                    break
                    
                item = self.embed_queue.get()
                if item is None:  # End signal
                    logger.info("Uploader worker received end signal")
                    self.embed_queue.task_done()
                    break
                    
                file_path, chunks, vectors = item
                try:
                    # Start with configured batch size
                    current_batch_size = self.config.batch_size
                    for i in range(0, len(chunks), current_batch_size):
                        batch = chunks[i:i + current_batch_size]
                        batch_count += 1
                        batch_ids = [self._make_deterministic_id(doc, i + j) for j, doc in enumerate(batch)]
                        
                        try:
                            logger.info(f"Uploading batch {batch_count} from {file_path} ({len(batch)} chunks)")
                            vector_store.add_documents(documents=batch, ids=batch_ids)
                            logger.info(f"Batch {batch_count} uploaded successfully")
                            self.chunks_uploaded += len(batch)
                            self._print_progress()
                        except Exception as e:
                            if "Request size" in str(e) and "exceeds the maximum supported size" in str(e):
                                logger.warning(f"Batch too large ({current_batch_size}), reducing size and retrying...")
                                current_batch_size = current_batch_size // 2
                                if current_batch_size < 1:
                                    raise Exception("Cannot reduce batch size further")
                                i -= current_batch_size  # Retry the same batch with smaller size
                                continue
                            else:
                                raise
                except Exception as e:
                    logger.error(f"Error uploading batch from {file_path}: {str(e)}")
                    self._handle_error(e, "uploader", f"Failed to upload {file_path}")
                finally:
                    self.embed_queue.task_done()
        except Exception as e:
            logger.error(f"Fatal error in uploader worker: {str(e)}")
            self._handle_error(e, "uploader", "Fatal error in uploader worker")
        finally:
            logger.info("Uploader worker finished")
            # Ensure we signal end of processing even if there's an error
            try:
                while not self.embed_queue.empty():
                    self.embed_queue.get()
                    self.embed_queue.task_done()
            except:
                pass
            self.embed_queue.put(None)  # Signal end of uploading

    def _make_deterministic_id(self, doc: Any, chunk_index: int) -> str:
        """Generate a deterministic ID for a document chunk."""
        content = doc.page_content
        source = doc.metadata.get("source", "")
        base = f"{Path(source).as_posix()}-{chunk_index}-{content.strip()}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()

    def process_documents(self, directory: str, glob_patterns: str) -> None:
        """Process and upload documents using a pipelined approach."""
        try:
            # Get all matching files
            patterns = [p.strip() for p in glob_patterns.split(',')]
            all_files = []
            for pattern in patterns:
                matching_files = list(Path(directory).glob(pattern))
                all_files.extend(matching_files)
            
            if not all_files:
                raise ValueError(f"No files found in {directory} matching patterns: {glob_patterns}")
            
            # Set total files once
            self.total_files = len(all_files)
            print(f"📁 Found {self.total_files} files to process")
            # Add blank lines for progress display
            print("\n")
            
            # Start worker threads
            chunker_thread = Thread(target=self._chunker_worker, daemon=True)
            embedder_thread = Thread(target=self._embedder_worker, daemon=True)
            uploader_thread = Thread(target=self._uploader_worker, daemon=True)
            
            chunker_thread.start()
            embedder_thread.start()
            uploader_thread.start()
            
            # Start reading in main thread
            self._reader_worker(all_files)
            
            # Wait for all queues to be processed
            self.read_queue.join()
            self.chunk_queue.join()
            self.embed_queue.join()
            
            # Print final newline after progress
            print("\n")
            
            # Check for errors
            if self.error_event.is_set():
                print("\n❌ Errors occurred during processing:")
                for error in self.errors:
                    print(f"\n{error}")
                raise Exception("Document processing failed due to errors in the pipeline")
            
            print("\n✅ All documents processed successfully!")
            
        except Exception as e:
            print(f"\n❌ Fatal error: {str(e)}")
            # Ensure we clean up any remaining items in queues
            for queue in [self.read_queue, self.chunk_queue, self.embed_queue]:
                try:
                    while not queue.empty():
                        queue.get()
                        queue.task_done()
                except:
                    pass
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


