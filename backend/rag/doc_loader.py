import re
import os
from typing import List
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_documents(path: str) -> List[Document]:
    """
    Load and chunk DOCX or PDF content. Uses Q-style split if applicable, else falls back to generic chunking.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(path)
        docs = loader.load()
        raw_text = "\n".join([doc.page_content for doc in docs])  # ✅ For PDFs
    elif ext == ".docx":
        loader = UnstructuredWordDocumentLoader(path, mode="single")
        docs = loader.load()
        raw_text = docs[0].page_content  # ✅ For DOCX (Unstructured returns 1 doc)
    else:
        raise ValueError("❌ Unsupported file format. Use .pdf or .docx only.")

    # Preview text
    print("📄 Preview of loaded text:\n", raw_text[:1000])

    # Chunking strategy
    chunks = split_by_question_pattern(raw_text)
    if len(chunks) > 1:
        print("🧠 Detected question-style format. Using Q-based chunking.")
        return [Document(page_content=chunk.strip()) for chunk in chunks if chunk.strip()]
    else:
        print("📄 No Q-patterns found. Using fallback text splitter.")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(raw_text)
        return [Document(page_content=chunk.strip()) for chunk in chunks]

def split_by_question_pattern(text: str) -> List[str]:
    """
    Split text on lines starting with question identifiers.
    """
    pattern = r"(?=\n?(?:Q[-\.\s]?|Question\s*)?\d{1,3}[\):\.\s-])"
    chunks = re.split(pattern, text)
    return [chunk for chunk in chunks if chunk.strip()]