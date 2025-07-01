# app_simple.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.messages import HumanMessage, AIMessage


load_dotenv()

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="deepseek/deepseek-chat-v3",
    streaming=True
)

# 🧠 Memory: List of message objects
chat_history = []

print("🤖 Streaming Chatbot with Memory (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Add user's message to history
    chat_history.append(HumanMessage(content=user_input))

    # Stream the bot's reply token-by-token
    print("Bot: ", end="", flush=True)
    reply_chunks = []
    for chunk in llm.stream(chat_history):
        print(chunk.content, end="", flush=True)
        reply_chunks.append(chunk.content)
    print()

    # Add bot's full response to history
    full_response = "".join(reply_chunks)
    chat_history.append(AIMessage(content=full_response))
