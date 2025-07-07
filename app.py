# app_simple.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from rag.retriever_chain import get_rag_chain
from langchain.memory import ConversationBufferMemory

load_dotenv()

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="deepseek/deepseek-chat-v3",
    streaming=True,
)

# Set up memory for the RAG chain
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

rag_chain = get_rag_chain(llm, memory)

print("🤖 RAG Chatbot with Memory (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    result = rag_chain.invoke({"question": user_input})
    print("Bot:", result["answer"])

    # Optional: print source snippet

    # sources = result.get("source_documents", [])
    # if sources:
    #     print("📚 Source: This answer was retrieved from your document.")
    #     for doc in sources:
    #         snippet = doc.page_content[:150].replace("\n", " ").strip()
    #         print("→ Source snippet:", snippet + "...")
    # else:
    #     print("💡 Note: This answer was generated without using your document.")
