from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
from pinecone import Pinecone

load_dotenv()

print("🤖 Initializing chat engine...")

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Check if index exists
index_name = os.getenv("PINECONE_INDEX_NAME")
indexes = pc.list_indexes()
if index_name not in [ix.name for ix in indexes]:
    print(f"❌ Error: Index '{index_name}' not found in Pinecone!")
    exit(1)

# Get index stats
index = pc.Index(index_name)
stats = index.describe_index_stats()
print(f"📊 Index stats: {stats}")

vector_store = PineconeVectorStore(
    index_name=index_name,
    api_key=os.getenv("PINECONE_API_KEY"),
)

index = VectorStoreIndex.from_vector_store(vector_store)
chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    streaming=True,          # streams tokens for snappy UX
)

print("✅ Chat engine ready!")
print("\n💬 Type your questions (type 'exit' to quit)")
print("----------------------------------------")

while True:
    user_input = input("\nYou: ").strip()
    
    if user_input.lower() in ['exit', 'quit', 'q']:
        print("\n👋 Goodbye!")
        break
        
    if not user_input:
        continue
        
    print("\nAssistant: ", end="", flush=True)
    try:
        response = chat_engine.chat(user_input)
        if not response.response:
            print("(No response generated. This might indicate that no relevant documents were found in the index.)")
        else:
            print(response.response)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
