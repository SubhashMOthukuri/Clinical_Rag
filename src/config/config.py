from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

def _required(key:str)-> str:
    """Get a required env var or failed fast at startup."""
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Missing required env var: {key}")
    return value

# API keys
OPENAI_API_KEY = _required("OPENAI_API_KEY")
GEMINI_API_KEY = _required("GEMINI_API_KEY")
GROQ_API_KEY = _required("GROQ_API_KEY")
PINECONE_API_KEY = _required("PINECONE_API_KEY")

# Config with defaults
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medreconcile-clinical-rag")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
