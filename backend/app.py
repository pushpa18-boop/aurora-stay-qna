import os
import sys

# ---------------------------------------------------------
# Safety Environment Variable Validation
# ---------------------------------------------------------
required_vars = ["OPENAI_API_KEY", "OPENAI_API_BASE"]
missing_vars = [var for var in required_vars if not os.getenv(var)]

if missing_vars:
    print(f"❌ CRITICAL STARTUP ERROR: Missing required environment variables: {missing_vars}", file=sys.stderr)
    print("Please set these in your GitHub Codespace Secrets and rebuild/restart the container!", file=sys.stderr)

# Import necessary libraries
from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Import RAG utility functions
from rag_utils import retrieve, rag

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
EMBED_MODEL = "HarishMaths/Hotel-Policy-Embedding"
VECTORSTORE_PATH = "faiss_index"

# ---------------------------------------------------------
# Initialize the Flask application
# ---------------------------------------------------------
rag_api = Flask("Aurora Stay API")

# ---------------------------------------------------------
# Load the embedding model
# ---------------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# ---------------------------------------------------------
# Load the saved FAISS vector store
# ---------------------------------------------------------
vectorstore = FAISS.load_local(
    VECTORSTORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

print(f"Vectors stored: {vectorstore.index.ntotal}")

# ---------------------------------------------------------
# Default route
# ---------------------------------------------------------
@rag_api.get("/")
def home():
    return "Welcome to the Aurora Stay RAG API!"

# ---------------------------------------------------------
# Relevant chunks endpoint
# ---------------------------------------------------------
@rag_api.post("/v1/relevant_chunks")
def relevant_chunks():
    data = request.get_json()
    query = data["query"]
    k = data.get("k", 5)

    chunks = retrieve(
        query=query,
        k=k,
        vectorstore=vectorstore
    )

    return jsonify({
        "query": query,
        "k": k,
        "relevant_chunks": chunks
    })

# ---------------------------------------------------------
# Answer with relevant chunks endpoint
# ---------------------------------------------------------
@rag_api.post("/v1/answer_with_relevant_chunks")
def answer_with_relevant_chunks():
    if not os.getenv("OPENAI_API_KEY"):
        return jsonify({
            "error": "OPENAI_API_KEY environment variable is not configured on the backend server. Please verify your Codespace secrets."
        }), 500

    data = request.get_json()
    query = data["query"]
    k = data.get("k", 5)
    model_name = data.get("model_name", "gpt-4o-mini")
    temperature = data.get("temperature", 0.0)
    top_p = data.get("top_p", 1.0)
    max_tokens = data.get("max_tokens", 512)

    answer, chunks = rag(
        query=query,
        k=k,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        vectorstore=vectorstore
    )

    return jsonify({
        "query": query,
        "k": k,
        "model_name": model_name,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "relevant_chunks": chunks,
        "answer": answer
    })

# ---------------------------------------------------------
# Answer endpoint
# ---------------------------------------------------------
@rag_api.post("/v1/answer")
def answer():
    if not os.getenv("OPENAI_API_KEY"):
        return jsonify({
            "error": "OPENAI_API_KEY environment variable is not configured on the backend server."
        }), 500

    data = request.get_json()
    query = data["query"]
    k = data.get("k", 5)
    model_name = data.get("model_name", "gpt-4o-mini")
    temperature = data.get("temperature", 0.0)
    top_p = data.get("top_p", 1.0)
    max_tokens = data.get("max_tokens", 512)

    generated_answer, _ = rag(
        query=query,
        k=k,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        vectorstore=vectorstore
    )

    return jsonify({
        "answer": generated_answer
    })

if __name__ == "__main__":
    rag_api.run(debug=True)