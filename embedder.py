# app.py
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, Settings, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from pinecone import Pinecone
from openai import OpenAI

load_dotenv()

# Set embed model and node parser
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large", dimensions=3072)
Settings.node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=100)

# Confirm embedding model dimensions
vector = Settings.embed_model.get_text_embedding("Test this text.")
print(f"Embedding dimension: {len(vector)}")  # should be 3072

# Direct test using the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
response = client.embeddings.create(
    input="Test this embedding",
    model="text-embedding-3-large"
)
print(f"✅ OpenAI returned vector of dim: {len(response.data[0].embedding)}")

# 1️⃣  Load + chunk docs
try:
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
    
    print("📂 Scanning for documents...")
    # Load documents with specific file types
    docs = SimpleDirectoryReader(
        input_dir="docs",
        recursive=True,
        required_exts=['.pdf', '.docx', '.xlsx', '.txt']
    ).load_data()
    
    if not docs:
        print("❌ No documents were loaded successfully!")
        exit(1)
    
    # Convert to Document objects if needed
    print(f"📄 Processing {len(docs)} documents...")
    docs = [Document(text=doc.text, metadata=doc.metadata) if not isinstance(doc, Document) else doc for doc in docs]
    
    print(f"✅ Successfully loaded {len(docs)} documents")
    print("\n📊 Document types found:")
    doc_types = {}
    for doc in docs:
        file_type = os.path.splitext(doc.metadata.get('file_name', ''))[1]
        doc_types[file_type] = doc_types.get(file_type, 0) + 1
    for file_type, count in doc_types.items():
        print(f"  - {file_type}: {count} files")

except Exception as e:
    print(f"Error loading documents: {str(e)}")
    exit(1)

# 2️⃣  Connect to Pinecone
print("\n🔌 Connecting to Pinecone...")
# Initialize Pinecone client
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

# Check if index exists
index_name = os.getenv("PINECONE_INDEX_NAME")
indexes = pc.list_indexes()
if index_name not in [ix.name for ix in indexes]:
    print(f"❌ Error: Index '{index_name}' not found in Pinecone!")
    exit(1)

# Get index stats before upload
index = pc.Index(index_name)
initial_stats = index.describe_index_stats()
print(f"📊 Initial index stats: {initial_stats}")

vector_store = PineconeVectorStore(
    index_name=index_name,
    api_key=os.getenv("PINECONE_API_KEY"),
    dimensions=3072
)

# Create storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create index
print("📚 Building index...")
index = VectorStoreIndex.from_documents(
    docs,
    embed_model=Settings.embed_model,
    storage_context=storage_context,
    show_progress=True  # Show progress bar
)

# Verify upload
final_stats = pc.Index(index_name).describe_index_stats()
print(f"\n📊 Final index stats: {final_stats}")

if final_stats['total_vector_count'] > initial_stats['total_vector_count']:
    print("✅ Documents successfully uploaded to Pinecone!")
else:
    print("❌ Warning: No new vectors were added to Pinecone. Please check the logs above for errors.")


