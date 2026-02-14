# Streamlit Cloud Setup (Single Path: Hugging Face)

## 1) Deploy app

- Repository: your GitHub repo
- Main file path: `app/streamlit_app.py`

## 2) Add Streamlit secrets

```toml
HF_API_BASE = "https://api-inference.huggingface.co/models"
HF_TOKEN = "your_hf_token"
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_CHAT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
CHROMA_COLLECTION = "plan_alignment_hf"
```

## 3) Reboot app

After saving secrets, reboot app from Streamlit settings.

## 4) Use the app

Upload CSV files and run analysis. Download generated outputs from the UI.

## 5) Input CSV schema

### Strategic Plan CSV
- `section_name`
- `text`

### Action Plan CSV
- `section_name`
- `text`

### Optional Ground Truth CSV
- `strategy_chunk_id`
- `matched_action_chunk_ids`
