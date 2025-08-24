# from logger_config import setup_logger

# logger = setup_logger()
from logger_config import setup_logger
from agent import RAGAgent
from chat_runtime import ChatRuntime

if __name__ == "__main__":
    setup_logger()
    agent = RAGAgent(knowledge_base="policies_db.json")  # compiled without persistence
    chat = ChatRuntime(agent)  # recompile with SQLite persistence

    thread_id = "aayush-local-2"
    print("Type 'reset' to clear history, 'export' to save, 'quit' to exit.")
    while True:
        user = input("You: ").strip()
        if user.lower() == "quit":
            break
        if user.lower() == "reset":
            thread_id = chat.reset_thread(thread_id)
            print("(history cleared)")
            continue
        if user.lower() == "export":
            path = chat.export_history(thread_id)
            print(f"(saved to {path})")
            continue
        reply = chat.chat(user, thread_id)
        print("Bot:", reply)
