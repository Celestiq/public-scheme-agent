import os, json, pickle, hashlib
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from datetime import datetime
import pytz
import logging
import time

from src.core.prompts import SYSTEM_PROMPT, TOOLS_POLICY
from src.core.logger_config import setup_logger

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain_core.tools import Tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, HumanMessage, message_to_dict
from langgraph.graph import START, END, StateGraph, MessagesState
from langchain_community.vectorstores import FAISS


load_dotenv()

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CACHE_DIR = Path("./rag_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RETRIEVE_K = 5

def _file_hash(path: str) -> str:
    """Stable key so cache is invalidated if source changes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    h.update(f"::cs={CHUNK_SIZE}::co={CHUNK_OVERLAP}".encode())
    return h.hexdigest()[:16]

class RAGAgent:
    def __init__(self, knowledge_base: str, checkpointer = None):
        self.logger = logging.getLogger("app.agent")
        self.logger.info("********************* Setting up RAG Agent... *********************")

        try:
            if not Path(knowledge_base).exists():
                self.logger.error("Knowledge base file not found: %s", knowledge_base)
                raise FileNotFoundError(f"KB not found: {knowledge_base}")

            self.knowledge_base = knowledge_base
            self.cache_key = _file_hash(knowledge_base)
            self.logger.debug("Computed cache key: %s", self.cache_key)

            t0 = time.perf_counter()
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            self.llm = init_chat_model("gpt-5-mini", model_provider="openai")
            self.loader = JSONLoader(file_path=knowledge_base, jq_schema=".[]", text_content=False)
            self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            self.logger.info("Models and loaders initialized in %.2f ms", (time.perf_counter() - t0) * 1000)

            self._chunks_pkl = CACHE_DIR / f"{self.cache_key}.pkl"
            self._faiss_dir = CACHE_DIR / f"faiss_{self.cache_key}"

            self.chunks = self._load_or_build_chunks()
            self.vector_store = self._load_or_build_vector_store()

            # Expose bound method as a tool (no self in schema)
            self.retrieve_tool = Tool.from_function(
                func=self.retrieve,
                name="retrieve",
                description="Retrieve info related to a query"
            )
            self.checkpointer = checkpointer

            try:
                self.logger.info("Building Graph...")
                t1 = time.perf_counter()
                self._build_graph()
                self.logger.info("Graph build successful in %.2f ms", (time.perf_counter() - t1) * 1000)
            except Exception:
                self.logger.exception("Graph build failed")
                raise

            self.logger.info("Initialization successful!")
        except Exception:
            self.logger.exception("Initialization failed")
            raise

    def _load_or_build_chunks(self) -> List[Document]:
        try:
            if self._chunks_pkl.exists():
                t0 = time.perf_counter()
                with open(self._chunks_pkl, "rb") as f:
                    chunks = pickle.load(f)
                self.logger.info("Loaded %d chunks from cache: %s (%.2f ms)",
                                 len(chunks), self._chunks_pkl, (time.perf_counter() - t0) * 1000)
                return chunks

            self.logger.info("Cache miss. Loading and splitting documents from %s", self.knowledge_base)
            t1 = time.perf_counter()
            docs = self.loader.load()
            if not docs:
                self.logger.warning("No documents loaded from %s", self.knowledge_base)
            splits = self.text_splitter.split_documents(docs)
            with open(self._chunks_pkl, "wb") as f:
                pickle.dump(splits, f)
            self.logger.info("Created and cached %d chunks at %s (%.2f ms)",
                             len(splits), self._chunks_pkl, (time.perf_counter() - t1) * 1000)
            return splits
        except Exception:
            self.logger.exception("Failed to load or build chunks")
            raise

    def _load_or_build_vector_store(self):
        try:
            if self._faiss_dir.exists():
                t0 = time.perf_counter()
                vs = FAISS.load_local(
                    folder_path=str(self._faiss_dir),
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self.logger.info("Loaded FAISS index from %s (%.2f ms)",
                                 self._faiss_dir, (time.perf_counter() - t0) * 1000)
                return vs

            self.logger.info("Building FAISS index (chunks=%d)", len(self.chunks))
            t1 = time.perf_counter()
            vs = FAISS.from_documents(self.chunks, self.embeddings)
            vs.save_local(str(self._faiss_dir))
            self.logger.info("Built and saved FAISS index to %s (%.2f ms)",
                             self._faiss_dir, (time.perf_counter() - t1) * 1000)
            return vs
        except Exception:
            self.logger.exception("Failed to load or build vector store")
            raise

    def retrieve(self, query: str) -> str:
        """
        Retrieve information related to a query.
        """
        self.logger.info("Retrieve called with k=%d | query=%r", RETRIEVE_K, query)
        try:
            t0 = time.perf_counter()
            retrieved_docs = self.vector_store.similarity_search(query, k=RETRIEVE_K)
            dt = (time.perf_counter() - t0) * 1000
            self.logger.info("Retrieved %d docs in %.2f ms", len(retrieved_docs), dt)

            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\nContent: {doc.page_content}")
                for doc in retrieved_docs
            )
            if not serialized:
                self.logger.warning("Retrieve returned empty serialization for query=%r", query)
            return serialized
        except Exception:
            self.logger.exception("Retrieve failed for query=%r", query)
            raise

    def query_or_respond(self, state: MessagesState):
        """
        Either query the knowledge base or respond directly
        """
        try:
            self.logger.debug("query_or_respond: messages=%d", len(state["messages"]))
            llm_with_tools = self.llm.bind_tools([self.retrieve_tool])
            t0 = time.perf_counter()
            response = llm_with_tools.invoke([SystemMessage(TOOLS_POLICY)] + state["messages"])
            self.logger.info("LLM invoke (query_or_respond) completed in %.2f ms",
                             (time.perf_counter() - t0) * 1000)
            return {"messages": [response]}
        except Exception:
            self.logger.exception("query_or_respond failed")
            raise

    def generate(self, state: MessagesState):
        """
        Generate a final answer
        """
        try:
            recent_tool_messages = []
            for message in reversed(state["messages"]):
                if message.type == "tool":
                    recent_tool_messages.append(message)
                else:
                    break
            tool_messages = recent_tool_messages[::-1]
            self.logger.debug("generate: found %d recent tool messages", len(tool_messages))

            docs_content = "\n\n".join(doc.content for doc in tool_messages)
            if not docs_content:
                self.logger.warning("generate: no tool content found; proceeding with conversation only")

            conversation_messages = [
                message
                for message in state["messages"]
                if message.type in ("human", "system")
                or (message.type == "ai" and not message.tool_calls)
            ]
            self.logger.debug("generate: conversation messages=%d", len(conversation_messages))

            prompt = [SystemMessage(SYSTEM_PROMPT + f"\nContext:\n{docs_content}")] + conversation_messages

            t0 = time.perf_counter()
            response = self.llm.invoke(prompt)
            self.logger.info("LLM invoke (generate) completed in %.2f ms", (time.perf_counter() - t0) * 1000)
            return {"messages": [response]}
        except Exception:
            self.logger.exception("generate failed")
            raise

    def _build_graph(self):
        try:
            graph_builder = StateGraph(MessagesState)
            tools = ToolNode([self.retrieve_tool])

            graph_builder.add_node(self.query_or_respond)
            graph_builder.add_node(tools)
            graph_builder.add_node(self.generate)

            graph_builder.set_entry_point("query_or_respond")
            graph_builder.add_conditional_edges(
                "query_or_respond",
                tools_condition,
                {END: END, "tools": "tools"}
            )
            graph_builder.add_edge("tools", "generate")
            graph_builder.add_edge("generate", END)

            self.graph = graph_builder.compile(checkpointer=self.checkpointer)
            # self.logger.debug("Graph compiled and ready")
            self.logger.info("Graph compiled%s",
                         " with persistence" if self.checkpointer else "")
        except Exception:
            self.logger.exception("_build_graph failed")
            raise