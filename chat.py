import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from dataclasses import dataclass
from typing import Optional, List, TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph
from langchain_core.messages import HumanMessage, AIMessage
import logging
from datetime import datetime
from pathlib import Path
import warnings

# Load environment variables at the start
load_dotenv(override=True)

# Set up logging
def setup_logging():
    """Configure logging to write to both file and console."""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"chat_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
        ]
    )
    
    # Configure warnings to be logged
    logging.captureWarnings(True)
    
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging()

# Redirect warnings to logger
def warning_handler(message, category, filename, lineno, file=None, line=None):
    logger.warning(f"{category.__name__}: {message}")

warnings.showwarning = warning_handler

@dataclass
class ChatConfig:
    """Configuration for the chat engine."""
    streaming: bool
    chat_mode: str
    model_name: str
    temperature: float
    embedding_model: str  # Add embedding model configuration

    @classmethod
    def from_env(cls) -> 'ChatConfig':
        """Create configuration from environment variables."""
        return cls(
            streaming=os.getenv("CHAT_STREAMING", "true").lower() == "true",
            chat_mode=os.getenv("CHAT_MODE"),
            model_name=os.getenv("CHAT_MODEL"),
            temperature=float(os.getenv("CHAT_TEMPERATURE")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "openai")  # Default to OpenAI if not specified
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
        logger.info("Setting up Pinecone connection...")
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        logger.info(f"Using Pinecone index: {self.index_name}")
        
        # Check if index exists
        indexes = self.pc.list_indexes()
        if self.index_name not in [ix.name for ix in indexes]:
            error_msg = f"Index '{self.index_name}' not found in Pinecone!"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Get index stats
        self.index = self.pc.Index(self.index_name)
        stats = self.index.describe_index_stats()
        logger.info(f"Index stats: {stats}")
        print(f"📊 Index stats: {stats}")

    def _setup_embeddings(self) -> None:
        """Set up the embedding model."""
        print("⚙️  Configuring embedding model...")
        logger.info(f"Setting up {self.config.embedding_model} embeddings...")
        
        if self.config.embedding_model.lower() == "openai":
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        elif self.config.embedding_model.lower() == "bge":
            self.embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-base-en-v1.5",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        else:
            error_msg = f"Unsupported embedding model: {self.config.embedding_model}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        logger.info(f"✅ {self.config.embedding_model} embeddings configured successfully")

    def _setup_vector_store(self) -> None:
        """Set up the vector store from existing index."""
        print("📚 Loading vector store...")
        logger.info("Setting up vector store...")
        self.vector_store = PineconeVectorStore.from_existing_index(
            index_name=self.index_name,
            embedding=self.embeddings
        )
        logger.info("Vector store configured successfully")

    def _setup_llm(self) -> None:
        """Set up the language model."""
        print("🤖 Configuring language model...")
        logger.info(f"Setting up language model: {self.config.model_name}")
        self.llm = ChatOpenAI(
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            streaming=self.config.streaming
        )
        logger.info("Language model configured successfully")

    def _setup_prompt(self) -> None:
        """Set up the chat prompt template."""
        logger.info("Setting up chat prompt template")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant. Use the following context to answer the user's question.
            If you don't know the answer, just say that you don't know. Don't try to make up an answer.
            
            Context: {context}
            
            Question: {question}"""),
            ("human", "{question}")
        ])
        logger.info("Chat prompt template configured")

    def _retrieve(self, state: State) -> dict:
        """Retrieve relevant documents for the question."""
        print("\n🔍 Retrieving relevant documents...")
        logger.info(f"Retrieving documents for question: {state['question'][:100]}...")
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        print(f"✅ Retrieved {len(retrieved_docs)} documents")
        return {"context": retrieved_docs}

    def _generate(self, state: State) -> dict:
        """Generate an answer based on the retrieved context."""
        print("\n💭 Generating answer...")
        logger.info("Generating answer from retrieved context")
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke({
            "question": state["question"],
            "context": docs_content
        })
        response = self.llm.invoke(messages)
        logger.info("Answer generated successfully")
        return {"answer": response.content}

    def _setup_graph(self) -> None:
        """Set up the LangGraph workflow."""
        print("🔄 Setting up chat workflow...")
        logger.info("Setting up LangGraph workflow")
        graph_builder = StateGraph(State)
        
        # Add nodes
        graph_builder.add_node("retrieve", self._retrieve)
        graph_builder.add_node("generate", self._generate)
        
        # Add edges
        graph_builder.add_edge(START, "retrieve")
        graph_builder.add_edge("retrieve", "generate")
        
        # Compile the graph
        self.graph = graph_builder.compile()
        logger.info("Chat workflow configured successfully")
        print("✅ Chat workflow ready!")

    def chat(self, question: str) -> Optional[str]:
        """Process a user input and return the response."""
        if not question.strip():
            return None

        try:
            logger.info(f"Processing question: {question[:100]}...")
            # Stream the steps for better visibility
            for step in self.graph.stream(
                {"question": question},
                stream_mode="updates"
            ):
                if "retrieve" in step:
                    print("\n📚 Retrieved documents:")
                    for doc in step["retrieve"]["context"]:
                        source = doc.metadata.get('source', 'Unknown source')
                        print(f"- {source}")
                        logger.info(f"Retrieved document: {source}")
                elif "generate" in step:
                    print("\n✨ Generated answer:")
                    print(step["generate"]["answer"])
                    logger.info("Answer generated and displayed to user")

            # Get the final result
            result = self.graph.invoke({"question": question})
            logger.info("Chat completed successfully")
            return result["answer"]
        except Exception as e:
            error_msg = f"Error in chat: {str(e)}"
            logger.error(error_msg)
            print(f"\n❌ Error: {str(e)}")
            return None

def main():
    """Main entry point for the chat application."""
    # Create configuration from environment variables
    config = ChatConfig.from_env()
    logger.info("Chat application starting")

    # Create and initialize the chat engine
    chat_engine = ChatEngine(config)

    print("\n💬 Type your questions (type 'exit' to quit)")
    print("----------------------------------------")

    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            logger.info("User exited the chat")
            print("\n👋 Goodbye!")
            break
            
        print("\nAssistant: ", end="", flush=True)
        response = chat_engine.chat(user_input)
        if response:
            print(response)

if __name__ == "__main__":
    main()
