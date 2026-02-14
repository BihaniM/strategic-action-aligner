# Deployment Guide (Streamlit Cloud + Hugging Face)

This project deploys directly on Streamlit Cloud and uses Hugging Face Inference API.

## 1) Push code to GitHub

Ensure these files are committed:
- `app/streamlit_app.py`
- `requirements.txt`
- `runtime.txt`
- `src/` modules

## 2) Create Streamlit Cloud app

1. Open Streamlit Community Cloud
2. Select your repository
3. Set entrypoint to:

```text
app/streamlit_app.py
```

## 3) Configure Secrets

In Streamlit app settings, add:

```toml
HF_API_BASE = "https://api-inference.huggingface.co/models"
HF_TOKEN = "your_hf_token"
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_CHAT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
CHROMA_COLLECTION = "plan_alignment_hf"
```

## 4) Redeploy

- Save secrets
- Reboot app

## 5) Validate deployment

Upload:
- `strategic_plan.csv`
- `action_plan.csv`

Confirm:
- Overall alignment score appears
- Strategy table is rendered
- Low-alignment recommendations are produced
- Download buttons work

## Notes

- No local model runtime is required.
- No tunneling setup is required.
- All model inference is handled by Hugging Face APIs.
