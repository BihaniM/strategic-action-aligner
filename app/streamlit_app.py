import json
import os

import pandas as pd
import streamlit as st

_env_keys = [
    "CHROMA_COLLECTION",
    "OLLAMA_BASE_URL",
    "OLLAMA_EMBEDDING_MODEL",
    "OLLAMA_CHAT_MODEL",
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
from src.evaluation import evaluate_strategy_action_matching
from src.improvement_agent import run_improvement_agent_loop
from src.ingestion_pipeline import embed_and_store_chunks


st.set_page_config(page_title="Plan Alignment Analyzer", layout="wide")
st.title("Strategic vs Action Plan Alignment Analyzer")
st.caption("Upload strategy and action plans, compute alignment, and generate AI improvement suggestions.")

strategic_file = st.file_uploader("Upload Strategic Plan CSV", type=["csv"], key="strategic_csv")
action_file = st.file_uploader("Upload Action Plan CSV", type=["csv"], key="action_csv")
ground_truth_file = st.file_uploader(
    "Optional: Upload Ground Truth CSV (strategy_chunk_id, action_chunk_id)",
    type=["csv"],
    key="ground_truth_csv",
)

top_k = st.slider("Top matching actions per strategy", min_value=1, max_value=10, value=3, step=1)
low_alignment_threshold = st.slider(
    "Low-alignment threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    step=0.01,
)
persist_to_chroma = st.checkbox("Store uploaded chunks in ChromaDB", value=True)

if st.button("Run Alignment Analysis"):
    if strategic_file is None or action_file is None:
        st.error("Please upload both Strategic Plan and Action Plan CSV files.")
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

    st.subheader("Overall Alignment Score")
    st.metric("Alignment", f"{overall_alignment_score:.2f}%")

    st.subheader("Strategy-wise Alignment Table")
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

    st.subheader("Low-alignment Warnings")
    if low_alignment_df.empty:
        st.success("No low-alignment strategies found at the selected threshold.")
        improved_df = low_alignment_df
        suggestion_history = []
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

        st.subheader("AI-generated Improvement Suggestions")
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

        st.subheader("Post-improvement Low-alignment Table")
        st.dataframe(
            improved_df[["strategy", "matched_actions", "similarity_score", "section_name"]],
            use_container_width=True,
        )

    if ground_truth_file is not None:
        st.subheader("Alignment Evaluation (Ground Truth)")
        ground_truth_df = pd.read_csv(ground_truth_file)
        metrics = evaluate_strategy_action_matching(
            predicted_df=strategy_alignment_df,
            ground_truth_source=ground_truth_df,
        )
        st.dataframe(pd.DataFrame([metrics]), use_container_width=True)

        st.caption(
            f"Evaluated on {int(metrics['sample_size'])} labeled strategy-action pairs."
        )

    st.success("Alignment analysis completed successfully.")
