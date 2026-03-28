"""
server.py — Flask wrapper around the existing RAG pipeline.
Run with:  python3 server.py   (ensure Ollama is running)
"""
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from dotenv import load_dotenv
load_dotenv()

import glob
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os

# ── Document loading ───────────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader

pdf_files = glob.glob("data/*.pdf")
documents = []
for pdf in pdf_files:
    loader = PyPDFLoader(pdf)
    documents.extend(loader.load())

# ── Chunking ───────────────────────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
document_chunks = splitter.split_documents(documents)

# ── Embeddings (Ollama — nomic-embed-text) ─────────────────────────
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    keep_alive="-1",
)

# ── FAISS vector store ─────────────────────────────────────────────
from langchain_community.vectorstores import FAISS

FAISS_PATH = os.path.join(BASE_DIR, "faiss_index")

if os.path.exists(FAISS_PATH):
    #  Load from disk — fast, skips re-embedding
    print(" Loading existing FAISS index...")
    vector_store = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    # Build fresh and save for next time
    print(" Building FAISS index from scratch...")
    vector_store = FAISS.from_documents(document_chunks, embeddings)
    vector_store.save_local(FAISS_PATH)
    print("Index saved to disk.")
    
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)

# ── LLM (Ollama — Mistral) ────────────────────────────────────────
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = OllamaLLM(model="mistral", num_predict=264, temperature=0.5)

prompt = ChatPromptTemplate.from_template(
    """You are a detailed research assistant. Use the following pieces of retrieved context to provide a 
comprehensive, long-form answer to the question. 

If the context contains multiple details about the person, combine them into a narrative. 
If you don't know the answer based on the context, just say so—don't make it up.

Context:
{context}

Question: {question}

Answer:"""
)

qa_chain = (
    {"context": retriever, "question": lambda x: x}
    | prompt
    | llm
    | StrOutputParser()
)

# ── Flask app ──────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("message", "")
    if not query.strip():
        return jsonify({"response": "Please provide a message."}), 400
    try:
        response = qa_chain.invoke(query)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"}), 500


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "chatbox.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("🚀 RAG server running on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
