# Streamlit App Running Setup (Hugging Face)

## Local run

1. Install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Configure env:

```bash
copy .env.example .env
```

Set at least:
- `HF_TOKEN`

3. Start app:

```bash
streamlit run app/streamlit_app.py
```

## Streamlit Cloud run

Set these secrets:

```toml
HF_API_BASE = "https://api-inference.huggingface.co/models"
HF_TOKEN = "your_hf_token"
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_CHAT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
CHROMA_COLLECTION = "plan_alignment_hf"
```

Reboot app after updating secrets.
