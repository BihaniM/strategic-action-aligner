# Step-by-Step: Run Dashboard Locally (Hugging Face)

## 1) Prepare environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Update `.env` with your Hugging Face token:
- `HF_TOKEN`

## 2) Start Streamlit

```bash
streamlit run app/streamlit_app.py
```

## 3) Upload files in UI

Required:
- Strategic plan CSV (`section_name`, `text`)
- Action plan CSV (`section_name`, `text`)

Optional:
- Ground truth CSV (`strategy_chunk_id`, `matched_action_chunk_ids`)

## 4) Run analysis

Click **Run Alignment Analysis**.

You will get:
- Overall alignment %
- Strategy-wise match table
- Low-alignment list
- AI improvement suggestions
- Agentic recommendations
- Optional evaluation metrics

## 5) Troubleshooting

### Error: cannot connect to Hugging Face
- Verify `HF_TOKEN` is set
- Verify internet connectivity
- Verify model IDs in `.env`

### Error: unauthorized
- Regenerate HF token with inference access
- Replace `HF_TOKEN` and rerun

### Slow first request
- First call may be cold-started on Hugging Face
- Retry once if needed
