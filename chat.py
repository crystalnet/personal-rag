import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from dataclasses import dataclass
from typing import Optional, List, TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables at the start
load_dotenv(override=True)

@dataclass
class ChatConfig:
    """Configuration for the chat engine."""
    streaming: bool
    chat_mode: str
    model_name: str
    temperature: float

    @classmethod
    def from_env(cls) -> 'ChatConfig':
        """Create configuration from environment variables."""
        return cls(
            streaming=os.getenv("CHAT_STREAMING", "true").lower() == "true",
            chat_mode=os.getenv("CHAT_MODE"),
            model_name=os.getenv("CHAT_MODEL"),
            temperature=float(os.getenv("CHAT_TEMPERATURE"))
        )

class State(TypedDict):
    """State for the chat application."""
    question: str
    context: List[Document]
    answer: str

class ChatEngine:
    def __init__(self, config: ChatConfig = None):
        """Initialize the chat engine with configuration."""
        self.config = config or ChatConfig.from_env()
        self._setup_pinecone()
        self._setup_embeddings()
        self._setup_vector_store()
        self._setup_llm()
        self._setup_prompt()
        self._setup_graph()

    def _setup_pinecone(self) -> None:
        """Set up Pinecone connection and validate index."""
        print("🔌 Connecting to Pinecone...")
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        
        # Check if index exists
        indexes = self.pc.list_indexes()
        if self.index_name not in [ix.name for ix in indexes]:
            raise ValueError(f"Index '{self.index_name}' not found in Pinecone!")
        
        # Get index stats
        self.index = self.pc.Index(self.index_name)
        stats = self.index.describe_index_stats()
        print(f"📊 Index stats: {stats}")

    def _setup_embeddings(self) -> None:
        """Set up the embedding model."""
        print("⚙️  Configuring embedding model...")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    def _setup_vector_store(self) -> None:
        """Set up the vector store from existing index."""
        print("📚 Loading vector store...")
        self.vector_store = PineconeVectorStore.from_existing_index(
            index_name=self.index_name,
            embedding=self.embeddings
        )

    def _setup_llm(self) -> None:
        """Set up the language model."""
        print("🤖 Configuring language model...")
        self.llm = ChatOpenAI(
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            streaming=self.config.streaming
        )

    def _setup_prompt(self) -> None:
        """Set up the chat prompt template."""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant. Use the following context to answer the user's question.
            If you don't know the answer, just say that you don't know. Don't try to make up an answer.
            
            Context: {context}
            
            Question: {question}"""),
            ("human", "{question}")
        ])

    def _retrieve(self, state: State) -> dict:
        """Retrieve relevant documents for the question."""
        print("\n🔍 Retrieving relevant documents...")
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        print(f"✅ Retrieved {len(retrieved_docs)} documents")
        return {"context": retrieved_docs}

    def _generate(self, state: State) -> dict:
        """Generate an answer based on the retrieved context."""
        print("\n💭 Generating answer...")
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke({
            "question": state["question"],
            "context": docs_content
        })
        response = self.llm.invoke(messages)
        return {"answer": response.content}

    def _setup_graph(self) -> None:
        """Set up the LangGraph workflow."""
        print("🔄 Setting up chat workflow...")
        graph_builder = StateGraph(State)
        
        # Add nodes
        graph_builder.add_node("retrieve", self._retrieve)
        graph_builder.add_node("generate", self._generate)
        
        # Add edges
        graph_builder.add_edge(START, "retrieve")
        graph_builder.add_edge("retrieve", "generate")
        
        # Compile the graph
        self.graph = graph_builder.compile()
        print("✅ Chat workflow ready!")

    def chat(self, question: str) -> Optional[str]:
        """Process a user input and return the response."""
        if not question.strip():
            return None

        try:
            # Stream the steps for better visibility
            for step in self.graph.stream(
                {"question": question},
                stream_mode="updates"
            ):
                if "retrieve" in step:
                    print("\n📚 Retrieved documents:")
                    for doc in step["retrieve"]["context"]:
                        print(f"- {doc.metadata.get('source', 'Unknown source')}")
                elif "generate" in step:
                    print("\n✨ Generated answer:")
                    print(step["generate"]["answer"])

            # Get the final result
            result = self.graph.invoke({"question": question})
            return result["answer"]
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            return None

def main():
    """Main entry point for the chat application."""
    # Create configuration from environment variables
    config = ChatConfig.from_env()

    # Create and initialize the chat engine
    chat_engine = ChatEngine(config)

    print("\n💬 Type your questions (type 'exit' to quit)")
    print("----------------------------------------")

    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("\n👋 Goodbye!")
            break
            
        print("\nAssistant: ", end="", flush=True)
        response = chat_engine.chat(user_input)
        if response:
            print(response)

if __name__ == "__main__":
    main()
