from langchain.chains import ConversationalRetrievalChain
from rag.vector_store import load_vectorstore

def get_rag_chain(llm, memory):
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever()

    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer",  # ✅ This line is **missing** or not applied
    )

    print("[✅] FAISS index loaded. RAG chain is ready.")
    return rag_chain
