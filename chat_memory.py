# chat_memory.py
import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

session_histories = {}

def get_chat_runnable():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("❌ Missing OPENROUTER_API_KEY in .env")

    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model="deepseek/deepseek-chat-v3",
        streaming=True  # ✅ Enable streaming
    )

    def get_history(session_id: str):
        if session_id not in session_histories:
            session_histories[session_id] = ChatMessageHistory()
        return session_histories[session_id]

    return RunnableWithMessageHistory(
        RunnableLambda(lambda x: llm.invoke(x["input"])),
        get_history,
        input_messages_key="input"
    ), llm
