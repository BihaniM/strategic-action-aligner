import os

from dotenv import load_dotenv

load_dotenv()

CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "plan_alignment_hf")
HF_API_BASE = os.getenv("HF_API_BASE", "https://api-inference.huggingface.co/models").rstrip("/")
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_CHAT_MODEL = os.getenv("HF_CHAT_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
