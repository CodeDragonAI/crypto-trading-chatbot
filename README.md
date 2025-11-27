# Trading Chatbot with LangChain + FastAPI

# Overview
This project is a Trading Assistant Chatbot built with LangChain, Ollama, and FastAPI.  
It loads trading concepts from a PDF, embeds them into a vector database (Chroma), and uses retrieval‑augmented generation (RAG) to answer user questions confidently and naturally — without referencing documents or sources.

# Tech Stack
Python 3.11
LangChain (core, community, text splitters)
Ollama (LLM + embeddings)
ChromaDB (vector store)
FastAPI (REST API)
Uvicorn (server)
PyPDFLoader (PDF ingestion)

# Project Structure


# Workflow
1. Load PDF (`Trading_Concepts_Master.pdf`) with `PyPDFLoader`.
2. Split text into chunks using `RecursiveCharacterTextSplitter`.
3. Embed chunks** with `OllamaEmbeddings`.
4. Store vectors in `Chroma`.
5. Retrieve context for user queries.
6. Generate answers with `Ollama` LLM using a custom system prompt.
7. Expose API via FastAPI (`/ask` endpoint).

# Usage

# 1. Install dependencies
```bash
pip install -r requirements.txt
