import os

from dotenv import load_dotenv

load_dotenv()

CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "plan_alignment_ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:1b")
