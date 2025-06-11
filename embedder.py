# app.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
import nltk
import ssl
import time
import hashlib
from pathlib import Path

load_dotenv()

# Configuration from environment variables with defaults
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "5"))

# Check if punkt is already downloaded
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

# 1️⃣  Load + chunk docs
try:
    print("📂 Scanning for documents...")
    # Load documents with specific file types
    loader = DirectoryLoader("docs/2024_INSEAD", glob="241108_Electives Selection2.xlsx")
    docs = loader.load()
    
    if not docs:
        print("❌ No documents were loaded successfully!")
        exit(1)
    
    print(f"✅ Successfully loaded {len(docs)} documents")
    print("\n📊 Document types found:")
    doc_types = {}
    for doc in docs:
        file_type = os.path.splitext(doc.metadata.get('source', ''))[1]
        doc_types[file_type] = doc_types.get(file_type, 0) + 1
        print(f"\nDocument content preview: {doc.page_content[:200]}...")  # Debug: Show content preview
    for file_type, count in doc_types.items():
        print(f"  - {file_type}: {count} files")

    # Configure text splitter
    print("\n✂️  Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    
    # Split documents into chunks
    chunked_docs = text_splitter.split_documents(docs)
    print(f"✅ Split documents into {len(chunked_docs)} chunks")
    print(f"   Using chunk size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP}")

except Exception as e:
    print(f"Error loading documents: {str(e)}")
    exit(1)

# 2️⃣  Connect to Pinecone
print("\n🔌 Connecting to Pinecone...")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

index_name = os.getenv("PINECONE_INDEX_NAME")
indexes = pc.list_indexes()
if index_name not in [ix.name for ix in indexes]:
    print(f"❌ Error: Index '{index_name}' not found in Pinecone!")
    exit(1)

# Get index stats before upload
index = pc.Index(index_name)
initial_stats = index.describe_index_stats()
print(f"📊 Initial index stats: {initial_stats}")

# 3️⃣  Create index (embeds + uploads)
print("⚙️  Configuring embedding model...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Create vector store
vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

# Add documents to the vector store
print("\n📤 Adding documents to vector store...")

def make_deterministic_id(doc, chunk_index):
    content = doc.page_content
    source = doc.metadata.get("source", "")
    # Normalize path and include chunk index to ensure uniqueness
    base = f"{Path(source).as_posix()}-{chunk_index}-{content.strip()}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()

# Generate deterministic IDs for each chunk
ids = [make_deterministic_id(doc, i) for i, doc in enumerate(chunked_docs)]
print(f"Generated {len(ids)} unique IDs for chunks")

try:
    vector_store.add_documents(documents=chunked_docs, ids=ids)
    print("✅ Documents added to vector store")
except Exception as e:
    print(f"❌ Error adding documents to vector store: {str(e)}")
    exit(1)

# Verify upload with retries
print("\n🔄 Verifying upload...")
max_retries = MAX_RETRIES
retry_delay = RETRY_DELAY  # seconds

print(f"   Using max retries: {max_retries}, delay: {retry_delay}s")

for attempt in range(max_retries):
    time.sleep(retry_delay)  # Wait for index to update
    final_stats = pc.Index(index_name).describe_index_stats()
    print(f"📊 Stats check {attempt + 1}/{max_retries}: {final_stats}")
    
    if final_stats['total_vector_count'] > initial_stats['total_vector_count']:
        print("✅ Documents successfully uploaded to Pinecone!")
        break
    elif attempt < max_retries - 1:
        print(f"⏳ Waiting for index to update... (attempt {attempt + 1}/{max_retries})")
    else:
        print("❌ Warning: No new vectors were added to Pinecone after multiple checks. Please verify the upload manually.")


