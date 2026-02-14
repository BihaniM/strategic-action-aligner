# Strategic-Action Alignment System (Hugging Face + ChromaDB + Streamlit)

This project analyzes alignment between a Strategic Plan and an Action Plan using:
- Hugging Face Inference API for embeddings and reasoning
- ChromaDB for vector storage
- Streamlit for the dashboard

## 1) What the system does

1. Loads `strategic_plan.csv` and `action_plan.csv`
2. Chunks content into paragraph/section units
3. Builds a unified chunk dataframe with:
   - `text`
   - `document_type`
   - `section_name`
4. Generates embeddings from Hugging Face
5. Computes similarity matrix and strategy-action matches
6. Calculates overall alignment percentage
7. Flags low-alignment strategies
8. Generates improvement suggestions (JSON)
9. Runs an agentic diagnose/propose/critique layer
10. Optionally evaluates precision/recall/F1 against ground truth

## 2) Required input format

### `strategic_plan.csv`
Required columns:
- `section_name`
- `text`

### `action_plan.csv`
Required columns:
- `section_name`
- `text`

### Optional `ground_truth.csv`
Required columns:
- `strategy_chunk_id`
- `matched_action_chunk_ids`

`matched_action_chunk_ids` can be comma-separated IDs.

## 3) Environment setup

1. Create virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Configure environment variables:

```bash
copy .env.example .env
```

Set in `.env`:
- `HF_TOKEN`
- `HF_EMBEDDING_MODEL`
- `HF_CHAT_MODEL`
- `CHROMA_COLLECTION`

## 4) Run full pipeline (CLI)

```bash
python -m src.full_pipeline \
  --strategic-csv data/input/strategic_plan.csv \
  --action-csv data/input/action_plan.csv \
  --output-dir data/output \
  --top-k 3 \
  --low-alignment-threshold 0.6 \
  --max-iterations 3
```

Outputs are written under `data/output/`.

## 5) Run Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Upload CSV files in the UI and run analysis.

## 6) Streamlit Cloud secrets

Set these in Streamlit Cloud secrets:

```toml
HF_API_BASE = "https://api-inference.huggingface.co/models"
HF_TOKEN = "your_hf_token"
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_CHAT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
CHROMA_COLLECTION = "plan_alignment_hf"
```

## 7) Project structure

- `app/streamlit_app.py` — Dashboard UI
- `src/config.py` — Environment configuration
- `src/hf_client.py` — Hugging Face API client
- `src/data_loader.py` — CSV loading and chunk dataframe creation
- `src/embedding_engine.py` — Embedding and similarity helpers
- `src/vector_store.py` — ChromaDB wrapper
- `src/alignment_engine.py` — Alignment artifacts and scoring
- `src/improvement_agent.py` — Improvement loop and suggestions
- `src/agentic_reasoning.py` — Diagnose/propose/critique layer
- `src/evaluation.py` — Precision/recall/F1 utilities
- `src/full_pipeline.py` — End-to-end CLI orchestrator
