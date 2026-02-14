import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_env_keys = [
    "CHROMA_COLLECTION",
    "HF_API_BASE",
    "HF_TOKEN",
    "HF_EMBEDDING_MODEL",
    "HF_CHAT_MODEL",
]
for key in _env_keys:
    if key in st.secrets and not os.getenv(key):
        os.environ[key] = str(st.secrets[key])

from src.alignment_engine import (
    build_alignment_artifacts_from_dataframes,
    build_strategy_alignment_table,
    calculate_overall_alignment_percentage,
    get_low_alignment_pairs,
)
from src.agentic_reasoning import run_agentic_reasoning_layer
from src.evaluation import evaluate_strategy_action_matching
from src.improvement_agent import run_improvement_agent_loop
from src.ingestion_pipeline import embed_and_store_chunks
from src.hf_client import HFClient


def check_huggingface_connectivity() -> None:
    if not os.getenv("HF_TOKEN"):
        raise ValueError("Missing HF_TOKEN secret or environment variable.")
    client = HFClient()
    client.embed_texts(["health check"])


st.set_page_config(page_title="Plan Alignment Analyzer", layout="wide")
st.markdown(
    """
    <style>
    .hero {
        padding: 1rem 1.2rem;
        border-radius: 14px;
        background: linear-gradient(120deg, #12355b, #1f6aa5);
        color: #ffffff;
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 1.6rem;
        font-weight: 700;
    }
    .hero p {
        margin: 0.35rem 0 0 0;
        opacity: 0.95;
        font-size: 0.95rem;
    }
    .section-title {
        font-weight: 650;
        color: #143d66;
        margin-top: 0.5rem;
        margin-bottom: 0.2rem;
    }
    .card {
        border-radius: 12px;
        padding: 0.9rem 1rem;
        border: 1px solid #e6ecf2;
        background: #f8fbff;
    }
    .card h4 {
        margin: 0;
        font-size: 0.9rem;
        color: #3f5f7c;
        font-weight: 600;
    }
    .card p {
        margin: 0.35rem 0 0;
        font-size: 1.35rem;
        font-weight: 700;
        color: #0c2d4d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>Strategic vs Action Plan Alignment Analyzer</h1>
      <p>Upload plans, score alignment, detect low-alignment gaps, and generate AI recommendations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown('<div class="section-title">Input Files</div>', unsafe_allow_html=True)
    upload_col_1, upload_col_2, upload_col_3 = st.columns([1, 1, 1])
    with upload_col_1:
        strategic_file = st.file_uploader("Upload Strategic Plan CSV", type=["csv"], key="strategic_csv")
    with upload_col_2:
        action_file = st.file_uploader("Upload Action Plan CSV", type=["csv"], key="action_csv")
    with upload_col_3:
        ground_truth_file = st.file_uploader(
            "Optional Ground Truth CSV", type=["csv"], key="ground_truth_csv"
        )

with st.container(border=True):
    st.markdown('<div class="section-title">Analysis Controls</div>', unsafe_allow_html=True)
    cfg_col_1, cfg_col_2, cfg_col_3 = st.columns([1, 1, 1])
    with cfg_col_1:
        top_k = st.slider("Top matching actions", min_value=1, max_value=10, value=3, step=1)
    with cfg_col_2:
        low_alignment_threshold = st.slider(
            "Low-alignment threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.01,
        )
    with cfg_col_3:
        persist_to_chroma = st.checkbox("Store chunks in ChromaDB", value=True)

run_clicked = st.button("Run Alignment Analysis", type="primary", use_container_width=True)

if run_clicked:
    if strategic_file is None or action_file is None:
        st.error("Please upload both Strategic Plan and Action Plan CSV files.")
        st.stop()

    try:
        check_huggingface_connectivity()
    except Exception:
        st.error(
            "Cannot connect to Hugging Face Inference API. Configure Streamlit secrets: HF_TOKEN, HF_EMBEDDING_MODEL, and HF_CHAT_MODEL."
        )
        st.stop()

    strategic_df = pd.read_csv(strategic_file)
    action_df = pd.read_csv(action_file)

    artifacts = build_alignment_artifacts_from_dataframes(strategic_df, action_df)
    overall_alignment_score = calculate_overall_alignment_percentage(artifacts)
    strategy_alignment_df = build_strategy_alignment_table(artifacts, top_k=top_k)
    low_alignment_df = get_low_alignment_pairs(strategy_alignment_df, threshold=low_alignment_threshold)

    if persist_to_chroma:
        stored_count = embed_and_store_chunks(artifacts.chunks_df)
        st.info(f"Stored {stored_count} chunks in ChromaDB.")

    score_col, strat_col, low_col = st.columns(3)
    with score_col:
        st.markdown(
            f"""
            <div class="card">
              <h4>Overall Alignment</h4>
              <p>{overall_alignment_score:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with strat_col:
        st.markdown(
            f"""
            <div class="card">
              <h4>Strategies Scored</h4>
              <p>{len(strategy_alignment_df)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with low_col:
        st.markdown(
            f"""
            <div class="card">
              <h4>Low-Alignment Items</h4>
              <p>{len(low_alignment_df)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.container(border=True):
        st.markdown('<div class="section-title">Strategy-wise Alignment</div>', unsafe_allow_html=True)
        st.dataframe(
            strategy_alignment_df[["strategy", "matched_actions", "similarity_score", "section_name"]],
            use_container_width=True,
        )

        st.download_button(
            label="Download strategy alignment table",
            data=strategy_alignment_df.to_csv(index=False).encode("utf-8"),
            file_name="strategy_alignment_table.csv",
            mime="text/csv",
        )

    st.markdown('<div class="section-title">Low-alignment Warnings</div>', unsafe_allow_html=True)
    if low_alignment_df.empty:
        st.success("No low-alignment strategies found at the selected threshold.")
        improved_df = low_alignment_df
        suggestion_history = []
        agentic_df = low_alignment_df
    else:
        st.warning(f"Found {len(low_alignment_df)} low-alignment strategies.")
        st.dataframe(
            low_alignment_df[["strategy", "matched_actions", "similarity_score", "section_name"]],
            use_container_width=True,
        )

        improved_df, suggestion_history = run_improvement_agent_loop(
            low_alignment_df=low_alignment_df,
            similarity_threshold=low_alignment_threshold,
            max_iterations=3,
        )

        st.markdown('<div class="section-title">AI-generated Improvement Suggestions</div>', unsafe_allow_html=True)
        if suggestion_history:
            suggestion_rows = []
            for item in suggestion_history:
                suggestion_rows.append(
                    {
                        "iteration": item["iteration"],
                        "strategy": item["strategy"],
                        "baseline_similarity_score": item["baseline_similarity_score"],
                        "improved_similarity_score": item["improved_similarity_score"],
                        "improved": item["improved"],
                        "suggestions_json": json.dumps(item["suggestions"], ensure_ascii=False),
                    }
                )

            suggestions_df = pd.DataFrame(suggestion_rows)
            st.dataframe(suggestions_df, use_container_width=True)
            st.download_button(
                label="Download AI suggestions JSONL",
                data="\n".join(suggestions_df["suggestions_json"].tolist()).encode("utf-8"),
                file_name="ai_suggestions.jsonl",
                mime="application/json",
            )

        st.markdown('<div class="section-title">Post-improvement Low-alignment Table</div>', unsafe_allow_html=True)
        st.dataframe(
            improved_df[["strategy", "matched_actions", "similarity_score", "section_name"]],
            use_container_width=True,
        )

        agentic_df, _ = run_agentic_reasoning_layer(
            low_alignment_df=improved_df,
        )

        st.markdown('<div class="section-title">Agentic AI Reasoning Recommendations</div>', unsafe_allow_html=True)
        st.dataframe(agentic_df, use_container_width=True)
        st.download_button(
            label="Download agentic recommendations",
            data=agentic_df.to_csv(index=False).encode("utf-8"),
            file_name="agentic_recommendations.csv",
            mime="text/csv",
        )

    if ground_truth_file is not None:
        st.subheader("Alignment Evaluation (Ground Truth)")
        ground_truth_df = pd.read_csv(ground_truth_file)
        try:
            metrics = evaluate_strategy_action_matching(
                predicted_df=strategy_alignment_df,
                ground_truth_source=ground_truth_df,
            )
            st.dataframe(pd.DataFrame([metrics]), use_container_width=True)
            st.caption(
                f"Evaluated on {int(metrics['sample_size'])} labeled strategy-action pairs."
            )
        except ValueError as exc:
            st.warning(str(exc))

    st.success("Alignment analysis completed successfully.")
