# ChatBot (RAG-Enhanced)

A command-line chatbot using DeepSeek, LangChain, and Retrieval-Augmented Generation (RAG).

## Features
- Chat with an AI model using DeepSeek via LangChain
- Retrieval-Augmented Generation: answers are grounded in your own documents
- Loads API key securely from a `.env` file
- Smart document ingestion: supports `.docx` (Word) and `.pdf` with robust chunking
- Detects and splits by question patterns (Q-1, Q.2, Q1:, etc.) for better retrieval

## Setup

1. **Clone the repository**
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   - For `.docx` support, the `unstructured` package and its optional docx dependencies are required (see requirements.txt).
3. **Set up your environment variables**
   - Create a `.env` file in the project root:
     ```env
     OPENROUTER_API_KEY=your_openrouter_api_key_here
     ```
   - (Or use your OpenAI API key if using OpenAI models)

## Document Ingestion & Vector DB Build

Before running the chatbot, you must build the vector database from your documents:

1. Place your source documents (e.g., `.docx` or `.pdf` files) in the `docs/` folder.
2. Run the following in a Python shell or script:
   ```python
   from rag.doc_loader import load_documents
   from rag.vector_store import create_vectorstore

   docs = load_documents("docs/InterviewQuestions.docx")
   create_vectorstore(docs)
   ```
   This will create a FAISS index in `faiss_index/`.
   **Re-run this step if you add or change documents.**

### How Document Chunking Works
- For `.docx` and `.pdf`, the loader previews the first 1000 characters.
- If your document uses question patterns (Q-1, Q.2, Q1:, etc.), it will chunk by question for best retrieval.
- If no Q-patterns are found, it falls back to a generic text splitter (500 chars, 50 overlap).

## Usage

Run the chatbot from your terminal:
```bash
python app.py
```
Type your message and press Enter. Type `exit` to stop the chatbot.

## File Overview
- `app.py`: Main chatbot application (now uses RAG pipeline)
- `rag/doc_loader.py`: Loads and splits documents into chunks, supports Q-based chunking
- `rag/vector_store.py`: Embeds and stores document chunks in a FAISS vector DB
- `rag/retriever_chain.py`: Orchestrates retrieval and LLM response
- `chat_memory.py`: (Optional) Conversation memory utilities
- `requirements.txt`: Python dependencies
- `.env`: Environment variables (not tracked by git)
- `docs/`: Place your source documents here

## Notes
- Make sure your `.env` file is not committed to version control (see `.gitignore`).
- This project is for educational/demo purposes.
- You can customize chunk size, overlap, and retriever settings in the code.
- For `.docx` support, see the [unstructured documentation](https://github.com/Unstructured-IO/unstructured) for any system requirements.
