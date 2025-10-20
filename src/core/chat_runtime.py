import json, logging
from pathlib import Path
from datetime import datetime
import pytz
import sqlite3
import time

from langchain_core.messages import HumanMessage, message_to_dict
from langgraph.checkpoint.sqlite import SqliteSaver
from src.core.agent import RAGAgent

class ChatRuntime:
    def __init__(self, agent: RAGAgent, db_file="chat_state.db", logger_name="app.chat"):
        self.logger = logging.getLogger(logger_name)
        if agent.checkpointer is None:
            conn = sqlite3.connect(db_file, check_same_thread=False)
            agent.checkpointer = SqliteSaver(conn)
            agent._build_graph()
        self.agent = agent
        self.logger.info("ChatRuntime initialized with DB file: %s", db_file)

    def chat(self, text: str, thread_id: str) -> str:
        cfg = {"configurable": {"thread_id": thread_id}}
        self.logger.info(f"Turn | thread={thread_id} | text={text}")
        result = self.agent.graph.invoke({
            "messages": [HumanMessage(content=text)],
        }, config=cfg)
        msgs = result["messages"]
        final_ai = next(
            (m for m in reversed(msgs) if m.type == "ai" and not getattr(m, "tool_calls", None)), msgs[-1]
            )
        return final_ai.content if hasattr(final_ai, "content") else str(final_ai)
    
    def get_history(self, thread_id: str):
        cfg = {"configurable": {"thread_id": thread_id}}
        snapshot = self.agent.graph.get_state(cfg)
        return snapshot.state.get("messages", [])

    def export_history(self, thread_id: str, out_dir="output") -> str:
        msgs = self.get_history(thread_id)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        ist = pytz.timezone("Asia/Kolkata")
        now = datetime.now(ist).strftime("%Y-%m-%d_%H-%M-%S")
        path = Path(out_dir) / f"{thread_id}_{now}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump([message_to_dict(m) for m in msgs], f, indent=2, ensure_ascii=False)
        self.logger.info("Exported history → %s", path)
        return str(path)

    # def reset_thread(self, thread_id: str):
    #     cfg = {"configurable": {"thread_id": thread_id}}
    #     self.agent.graph.delete_state(cfg)
    #     self.logger.info("History cleared for thread=%s", thread_id)

    def reset_thread(self, thread_id: str):
        cfg = {"configurable": {"thread_id": thread_id}}
        cp = getattr(self.agent, "checkpointer", None)

        # Preferred: ask the checkpointer to delete the thread’s checkpoints
        if cp and hasattr(cp, "delete"):
            cp.delete(cfg)   # removes all checkpoints for this thread_id
            self.logger.info("History cleared for thread=%s (checkpointer.delete)", thread_id)
            return thread_id

        # Fallback: start a fresh thread id (keeps old history intact)
        new_id = f"{thread_id}-{int(time.time())}"
        self.logger.warning(
            "Checkpointer has no 'delete'; using new thread_id=%s (old history retained).",
            new_id
        )
        return new_id
