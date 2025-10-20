import os
import uuid
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

# --- Local imports (support both your src.core layout and flat files) ---
try:
    from src.core.logger_config import setup_logger
    from src.core.agent import RAGAgent
    from src.core.chat_runtime import ChatRuntime
except Exception:  # fallback if running from repo root
    from logger_config import setup_logger  # type: ignore
    from agent import RAGAgent  # type: ignore
    from chat_runtime import ChatRuntime  # type: ignore

load_dotenv()
logger = setup_logger()

# ---- Configuration ----
KB_PATH = os.getenv("KB_PATH", "policies_db.json")

# Singletons to avoid reinitialization across requests
_runtime: ChatRuntime | None = None


def _ensure_runtime() -> ChatRuntime:
    global _runtime
    if _runtime is None:
        logger.info("Booting RAGAgent + ChatRuntime with KB=%s", KB_PATH)
        agent = RAGAgent(knowledge_base=KB_PATH)
        _runtime = ChatRuntime(agent)  # your SQLite-backed persistence
    return _runtime


# ---- Session / thread helpers ----

def _new_thread_id(session_hash: str | None) -> str:
    # New thread on each page load (or manual reset). Prefix with session hash for readability
    prefix = (session_hash or "session")[:16]
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# ---- Chat function (gr.ChatInterface contract) ----
# Accepts (message, history, thread_id) where thread_id comes from gr.State

def chat(message: str, history: list[dict], thread_id: str | None) -> str:
    try:
        runtime = _ensure_runtime()
        if not thread_id:
            # Fallback: generate a temporary thread if state was empty
            thread_id = _new_thread_id(None)
        logger.info("[thread=%s] User: %s", thread_id, message)
        reply = runtime.chat(message, thread_id)
        logger.info("[thread=%s] Assistant: %s", thread_id, reply)
        return reply
    except Exception as e:
        logger.exception("Chat error")
        return f"Sorry — something went wrong on my side: {type(e).__name__}: {e}"


# ---- Gradio UI ----

def build_ui() -> gr.Blocks:
    with gr.Blocks(fill_height=True, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Policy Assistant"
        )

        # Persistent per-tab state: current thread_id
        thread_state = gr.State(value=None)

        # Create the ChatInterface but feed our thread_state as an additional input
        chat_iface = gr.ChatInterface(
            fn=chat,
            additional_inputs=[thread_state],  # passed as third arg to `chat`
            # title="Policy Assistant",
            # description=(
            #     "Type your question. I search the knowledge base and respond."
            #     "Each page load starts a NEW thread; use the button below to reset manually."
            # ),
            chatbot=gr.Chatbot(height=500, avatar_images=(None, None)),
            textbox=gr.Textbox(placeholder="e.g., Am I eligible for any dairy subsidy in Karnataka?", scale=7),
            # retry_btn="Retry",
            # undo_btn="Undo",
            # clear_btn="Clear",  # clears the visible chat; thread continues until reset
        )

        with gr.Row():
            reset_btn = gr.Button("Start New Thread", variant="secondary")
            export_btn = gr.Button("Export Thread (JSON)")
            kb_info = gr.Markdown(f"**Knowledge base:** `{Path(KB_PATH).resolve()}`")

        # On page load: initialize a fresh thread bound to this browser session
        def _init_thread(request: gr.Request):
            runtime = _ensure_runtime()
            tid = _new_thread_id(getattr(request, "session_hash", None))
            logger.info("Initialized new thread: %s", tid)
            # Nothing to create server-side; ChatRuntime will lazily create rows on first write
            return tid

        demo.load(fn=_init_thread, inputs=None, outputs=thread_state)

        # Manual reset creates a brand-new thread id and clears the visible chat
        def _reset_thread(curr_tid: str | None, request: gr.Request):
            runtime = _ensure_runtime()
            tid = curr_tid or _new_thread_id(getattr(request, "session_hash", None))
            # Use your API if you want to explicitly clear: runtime.reset_thread(tid)
            try:
                runtime.reset_thread(tid)
            except Exception:
                # If your implementation returns a new id, you can assign it instead. For safety, just ignore.
                pass
            new_tid = _new_thread_id(getattr(request, "session_hash", None))
            logger.info("Reset thread %s -> %s", tid, new_tid)
            return new_tid, gr.update(value=[])

        reset_btn.click(_reset_thread, inputs=[thread_state], outputs=[thread_state, chat_iface.chatbot])

        # Export current thread via your runtime (expects a JSON path to be returned)
        def _export_thread(curr_tid: str | None):
            runtime = _ensure_runtime()
            tid = curr_tid or _new_thread_id(None)
            path = runtime.export_history(tid)
            return gr.File.update(value=path, visible=True)

        # A hidden File component to trigger the download UI after export
        downloader = gr.File(visible=False)
        export_btn.click(_export_thread, inputs=[thread_state], outputs=[downloader])

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch()