from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.alignment_engine import (
    build_alignment_artifacts,
    build_strategy_alignment_table,
    calculate_overall_alignment_percentage,
    get_low_alignment_pairs,
)
from src.agentic_reasoning import run_agentic_reasoning_layer
from src.evaluation import evaluate_strategy_action_matching
from src.improvement_agent import run_improvement_agent_loop
from src.ingestion_pipeline import embed_and_store_chunks


def run_full_pipeline(
    strategic_csv: str,
    action_csv: str,
    output_dir: str,
    low_alignment_threshold: float,
    top_k: int,
    max_iterations: int,
    persist_chroma: bool,
    ground_truth_csv: str | None = None,
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts = build_alignment_artifacts(
        strategic_csv_path=strategic_csv,
        action_csv_path=action_csv,
    )

    if persist_chroma:
        stored_count = embed_and_store_chunks(artifacts.chunks_df)
    else:
        stored_count = 0

    overall_alignment = calculate_overall_alignment_percentage(artifacts)
    strategy_table = build_strategy_alignment_table(artifacts, top_k=top_k)
    low_alignment = get_low_alignment_pairs(strategy_table, threshold=low_alignment_threshold)
    improved_low_alignment, improvement_history = run_improvement_agent_loop(
        low_alignment_df=low_alignment,
        similarity_threshold=low_alignment_threshold,
        max_iterations=max_iterations,
    )
    agentic_recommendations_df, agentic_reasoning_log = run_agentic_reasoning_layer(
        low_alignment_df=improved_low_alignment,
    )

    chunks_path = out_dir / "chunk_dataframe.csv"
    strategy_path = out_dir / "strategy_alignment_table.csv"
    low_path = out_dir / "low_alignment_pairs.csv"
    improved_low_path = out_dir / "improved_low_alignment_pairs.csv"
    suggestions_path = out_dir / "improvement_suggestions.jsonl"
    agentic_recommendations_path = out_dir / "agentic_recommendations.csv"
    agentic_reasoning_path = out_dir / "agentic_reasoning.jsonl"
    summary_path = out_dir / "pipeline_summary.json"

    artifacts.chunks_df.to_csv(chunks_path, index=False)
    strategy_table.to_csv(strategy_path, index=False)
    low_alignment.to_csv(low_path, index=False)
    improved_low_alignment.to_csv(improved_low_path, index=False)
    agentic_recommendations_df.to_csv(agentic_recommendations_path, index=False)

    with suggestions_path.open("w", encoding="utf-8") as f:
        for item in improvement_history:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with agentic_reasoning_path.open("w", encoding="utf-8") as f:
        for item in agentic_reasoning_log:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary: dict = {
        "overall_alignment_percentage": round(float(overall_alignment), 2),
        "stored_chunks_in_chromadb": int(stored_count),
        "strategy_rows": int(len(strategy_table)),
        "low_alignment_rows": int(len(low_alignment)),
        "improvement_records": int(len(improvement_history)),
        "agentic_recommendation_rows": int(len(agentic_recommendations_df)),
    }

    if ground_truth_csv:
        metrics = evaluate_strategy_action_matching(strategy_table, ground_truth_csv)
        summary["evaluation"] = metrics
        pd.DataFrame([metrics]).to_csv(out_dir / "evaluation_metrics.csv", index=False)

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full strategic-action alignment pipeline.")
    parser.add_argument("--strategic-csv", default="data/input/strategic_plan.csv")
    parser.add_argument("--action-csv", default="data/input/action_plan.csv")
    parser.add_argument("--output-dir", default="data/output")
    parser.add_argument("--low-alignment-threshold", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--no-chroma", action="store_true", help="Skip writing chunks to ChromaDB")
    parser.add_argument("--ground-truth-csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_full_pipeline(
        strategic_csv=args.strategic_csv,
        action_csv=args.action_csv,
        output_dir=args.output_dir,
        low_alignment_threshold=args.low_alignment_threshold,
        top_k=args.top_k,
        max_iterations=args.max_iterations,
        persist_chroma=not args.no_chroma,
        ground_truth_csv=args.ground_truth_csv,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
