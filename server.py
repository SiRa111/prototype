"""
server.py — Optimized Flask wrapper with SSE streaming and Cloud LLM/Ollama fallback.
"""
import os
import sys
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS

# ── Detect cloud keys and set up model/embeddings ─────────────────────
openai_key = os.environ.get("OPENAI_API_KEY")
gemini_key = os.environ.get("GEMINI_API_KEY")

if os.environ.get("VERCEL") and not gemini_key and not openai_key:
    print("⚠️ Running on Vercel without API keys. Defaulting to mock/dummy mode to prevent crash.")
    from langchain_core.embeddings import FakeEmbeddings
    from langchain_core.runnables import RunnableLambda
    embeddings = FakeEmbeddings(size=768)
    def _dummy_invoke(prompt):
        return "I'm running in lightweight local mode. Please configure your GEMINI_API_KEY or OPENAI_API_KEY in the Vercel dashboard."
    llm = RunnableLambda(_dummy_invoke)
    model_name = "dummy"
elif gemini_key:
    print("🌟 Running in GEMINI Cloud Mode")
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=gemini_key)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5, google_api_key=gemini_key)
    model_name = "gemini"
elif openai_key:
    print("🌟 Running in OPENAI Cloud Mode")
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, api_key=openai_key)
    model_name = "openai"
else:
    print("🌟 Running in LOCAL Ollama Mode (gemma:2b)")
    from langchain_ollama import OllamaEmbeddings, OllamaLLM
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    llm = OllamaLLM(model="gemma:2b", num_predict=256, temperature=0.5)
    model_name = "ollama"

# ── FAISS vector store ─────────────────────────────────────────────

from langchain_community.vectorstores import FAISS

FAISS_PATH = os.path.join(BASE_DIR, f"faiss_index_{model_name}")

if os.path.exists(FAISS_PATH):
    print(f" Loading existing FAISS index from {FAISS_PATH}...")
    vector_store = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    if os.environ.get("VERCEL"):
        print(f"⚠️ FAISS_PATH {FAISS_PATH} not found on Vercel. Creating a lightweight mock index to allow server boot...")
        from langchain_core.documents import Document
        fallback_docs = [Document(page_content="Error: The FAISS vector database is missing. Please compile the index locally and push it to Git.")]
        vector_store = FAISS.from_documents(fallback_docs, embeddings)
    else:
        print(f" Building FAISS index from scratch at {FAISS_PATH}...")
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        pdf_files = glob.glob(os.path.join(BASE_DIR, "data/*.pdf"))
        if not pdf_files:
            raise FileNotFoundError("No PDF files found in data/ directory!")
        
        documents = []
        for pdf in pdf_files:
            print(f"Loading {pdf}...")
            loader = PyPDFLoader(pdf)
            documents.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_chunks = splitter.split_documents(documents)

        # Filter out tiny junk chunks (section headers, blank pages, etc.)
        document_chunks = [c for c in all_chunks if len(c.page_content.strip()) > 80]
        print(f"   Kept {len(document_chunks)} of {len(all_chunks)} chunks (filtered short/empty ones)")

        import time
        batch_size = 50
        vector_store = None
        for i in range(0, len(document_chunks), batch_size):
            batch = document_chunks[i:i+batch_size]
            print(f"Embedding batch {i//batch_size + 1} of {(len(document_chunks) + batch_size - 1)//batch_size}...")
            try:
                if vector_store is None:
                    vector_store = FAISS.from_documents(batch, embeddings)
                else:
                    vector_store.add_documents(batch)
            except Exception as e:
                # Handle rate-limit with retry once
                print(f"Got error: {e}. Retrying in 65s...")
                time.sleep(65)
                if vector_store is None:
                    vector_store = FAISS.from_documents(batch, embeddings)
                else:
                    vector_store.add_documents(batch)
            if i + batch_size < len(document_chunks):
                print("Sleeping for 61s to respect API rate limits...")
                time.sleep(61)

        vector_store.save_local(FAISS_PATH)
        print("Index saved to disk.")
    
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 8},
)

# ── LLM Prompt and QA Chain ───────────────────────────────────────
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    """You are a helpful and detailed research assistant.
Use the following pieces of retrieved context to answer the question.
If the context is relevant, use it. If the context is not sufficient, you may also use your own general knowledge about the topic (such as the book 'Thinking, Fast and Slow' by Daniel Kahneman) to provide a complete, informative, and high-quality response.

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


@app.route("/chat/stream", methods=["GET"])
def chat_stream():
    query = request.args.get("message", "")
    if not query.strip():
        def error_gen():
            yield "data: Error: Please provide a message.\n\n"
        return Response(error_gen(), mimetype='text/event-stream')

    def generate():
        try:
            # Streams chunks of tokens from LLM chain
            for chunk in qa_chain.stream(query):
                # Clean or convert content if it's a ChatMessage chunk
                text_chunk = getattr(chunk, "content", chunk) if hasattr(chunk, "content") else str(chunk)
                yield f"data: {text_chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"

    return Response(generate(), mimetype='text/event-stream')


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(BASE_DIR, "chatbox.html")


@app.route("/assets/<path:path>", methods=["GET"])
def send_assets(path):
    return send_from_directory(os.path.join(BASE_DIR, "assets"), path)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": model_name})


@app.route("/api/tts", methods=["GET", "POST"])
def tts_edge():
    import asyncio
    import edge_tts

    if request.method == "POST":
        data = request.json or {}
        text = data.get("text", "")
        voice = data.get("voice", "en-US-AndrewNeural")
    else:
        text = request.args.get("text", "")
        voice = request.args.get("voice", "en-US-AndrewNeural")

    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    def generate():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        communicate = edge_tts.Communicate(text, voice, rate="-10%")

        async def get_chunks():
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]

        chunks = get_chunks()
        try:
            while True:
                chunk = loop.run_until_complete(chunks.__anext__())
                yield chunk
        except StopAsyncIteration:
            pass
        finally:
            loop.close()

    return Response(generate(), mimetype="audio/mpeg")


@app.route("/api/generate", methods=["POST"])
def generate_text():
    data = request.json or {}
    prompt = data.get("prompt", "")
    system_instruction = data.get("system_instruction", "")

    if not prompt.strip():
        return jsonify({"error": "No prompt provided"}), 400

    try:
        full_prompt = prompt
        if system_instruction:
            full_prompt = f"{system_instruction}\n\nCandidate / Context:\n{prompt}"

        response = llm.invoke(full_prompt)
        response_text = getattr(response, "content", str(response))
        return jsonify({"text": response_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(f"🚀 RAG server running on http://localhost:5000 ({model_name} mode)")
    app.run(host="0.0.0.0", port=5000, debug=False)
