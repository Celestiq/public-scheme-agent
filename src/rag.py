import os, json, pickle, hashlib
from pathlib import Path
from typing import List, TypedDict
from dotenv import load_dotenv
from time import time

from prompts import SYSTEM_PROMPT, HUMAN_PROMPT
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import START, StateGraph
from langchain_community.vectorstores import FAISS

load_dotenv()

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CACHE_DIR = Path("./rag_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def _file_hash(path: str) -> str:
    """Stable key so cache is invalidated if source changes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    h.update(f"::cs={CHUNK_SIZE}::co={CHUNK_OVERLAP}".encode())
    return h.hexdigest()[:16]

class RAGPipeline:
    def __init__(self, knowledge_base: str):
        print("Setting up...")
        self.knowledge_base = knowledge_base
        self.cache_key = _file_hash(knowledge_base)

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.llm = init_chat_model("gpt-5-nano", model_provider="openai")
        self.loader = JSONLoader(file_path=knowledge_base, jq_schema=".[]", text_content=False)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        self._chunks_pkl = CACHE_DIR / f"chunks_{self.cache_key}.pkl"
        self._faiss_dir = CACHE_DIR / f"faiss_{self.cache_key}"

        self.chunks = self._load_or_build_chunks()
        self.vector_store = self._load_or_build_vector_store()
        print("Initialization successful!")

        print("Building Graph...")
        self._build_graph()
        print("Graph build successful!")

    def _load_or_build_chunks(self) -> List[Document]:
        if self._chunks_pkl.exists():
            with open(self._chunks_pkl, "rb") as f:
                chunks = pickle.load(f)
            print(f"Loaded {len(chunks)} chunks from cache: {self._chunks_pkl}")
            return chunks

        docs = self.loader.load()
        splits = self.text_splitter.split_documents(docs)
        with open(self._chunks_pkl, "wb") as f:
            pickle.dump(splits, f)
        print(f"Created and cached {len(splits)} chunks at {self._chunks_pkl}")
        return splits

    def _load_or_build_vector_store(self):
        if self._faiss_dir.exists():
            vs = FAISS.load_local(
                folder_path=str(self._faiss_dir),
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True  # required for FAISS load
            )
            print(f"Loaded FAISS index from {self._faiss_dir}")
            return vs

        vs = FAISS.from_documents(self.chunks, self.embeddings)
        vs.save_local(str(self._faiss_dir))
        print(f"Built and saved FAISS index to {self._faiss_dir}")
        return vs

    def _query_vector_store(self, state: State):
        results = self.vector_store.similarity_search(state["question"])
        return {"context": results}
    
    def _generate_answer(self, state: State):
        docs_content = "\n\n".join([doc.page_content for doc in state["context"]])
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=HUMAN_PROMPT.format(context=docs_content, question=state["question"]))
        ]
        response = self.llm.invoke(messages)
        return {"answer": response.content}
    
    def _build_graph(self):
        graph_builder = StateGraph(State).add_sequence([self._query_vector_store, self._generate_answer])
        graph_builder.add_edge(START, "_query_vector_store")
        self.graph = graph_builder.compile()

    def invoke(self, question: str) -> str:
        response = self.graph.invoke({
            "question": question
        })
        response["context"] = "\n\n".join([doc.page_content for doc in response["context"]])
        with open(f"output/{time()}.json", "w") as f:
            json.dump(response, f, indent=2)
        
        return response["answer"]

if __name__ == "__main__":
    pipeline = RAGPipeline(knowledge_base="policies_db.json")
    question = "I have three children and live in Uttar Pradesh. Are there policies that can help me with their education?"
    answer = pipeline.invoke(question)
    print(answer)