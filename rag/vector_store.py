import os
from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Path to save/load the FAISS index
INDEX_DIR = "faiss_index/"
INDEX_FILE = os.path.join(INDEX_DIR, "index")


def create_vectorstore(docs):
    """
    Embeds docs and creates a FAISS vector store. Saves to disk.
    Args:
        docs (List[str] or List[Document]): List of text chunks or Document objects.
    Returns:
        FAISS: The created vector store.
    """
    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Check if index already exists
    if os.path.exists(INDEX_FILE + ".faiss") and os.path.exists(INDEX_FILE + ".pkl"):
        print("[ℹ️] Vector store already exists. Skipping creation.")
        return FAISS.load_local(INDEX_FILE, embeddings, allow_dangerous_deserialization=True)
    # Handle both Document and str
    if docs and isinstance(docs[0], Document):
        texts = [doc.page_content for doc in docs]
    else:
        texts = docs
    vectorstore = FAISS.from_texts(texts, embedding=embeddings)
    vectorstore.save_local(INDEX_FILE)
    print(f"[✅] Vector store saved to {INDEX_FILE}")
    return vectorstore


def load_vectorstore():
    """
    Loads the FAISS vector store from disk.
    Returns:
        FAISS: The loaded vector store.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(INDEX_FILE, embeddings, allow_dangerous_deserialization=True)
    return vectorstore
