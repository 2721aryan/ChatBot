# chat_memory.py
import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import ChatMessageHistory
from langchain_openai import ChatOpenAI  # ✅ Correct OpenAI class for base_url use

load_dotenv()

session_histories = {}

def get_chat_runnable():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("❌ Missing OPENROUTER_API_KEY in .env")

    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",  # ✅ OpenRouter-compatible
        api_key=api_key,
        model="deepseek/deepseek-chat-v3",
        streaming=True,
        temperature=0.7,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
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
