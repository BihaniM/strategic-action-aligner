from __future__ import annotations

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def evaluate_strategy_action_matching(
    predicted_df: pd.DataFrame,
    ground_truth_source,
) -> dict[str, float]:
    if isinstance(ground_truth_source, pd.DataFrame):
        ground_truth_df = ground_truth_source.copy()
    else:
        ground_truth_df = pd.read_csv(ground_truth_source)
    required_gt = {"strategy_chunk_id", "action_chunk_id"}
    missing_gt = required_gt.difference(ground_truth_df.columns)
    if missing_gt:
        raise ValueError(f"Ground truth CSV missing required columns: {sorted(missing_gt)}")

    required_pred = {"strategy_chunk_id", "matched_action_chunk_ids"}
    missing_pred = required_pred.difference(predicted_df.columns)
    if missing_pred:
        raise ValueError(f"Predicted dataframe missing required columns: {sorted(missing_pred)}")

    gt_map = {
        str(row.strategy_chunk_id): str(row.action_chunk_id)
        for row in ground_truth_df.itertuples(index=False)
    }

    y_true = []
    y_pred = []

    for row in predicted_df.itertuples(index=False):
        strategy_id = str(row.strategy_chunk_id)
        if strategy_id not in gt_map:
            continue
        pred_first = str(row.matched_action_chunk_ids).split("||")[0]
        y_true.append(gt_map[strategy_id])
        y_pred.append(pred_first)

    if not y_true:
        predicted_ids = set(predicted_df["strategy_chunk_id"].astype(str).tolist())
        ground_truth_ids = set(ground_truth_df["strategy_chunk_id"].astype(str).tolist())
        overlap_count = len(predicted_ids.intersection(ground_truth_ids))
        predicted_example = next(iter(predicted_ids), "N/A")
        ground_truth_example = next(iter(ground_truth_ids), "N/A")
        raise ValueError(
            "No overlapping strategy_chunk_id found between predictions and ground truth. "
            f"overlap_count={overlap_count}, "
            f"predicted_rows={len(predicted_ids)}, ground_truth_rows={len(ground_truth_ids)}, "
            f"predicted_example={predicted_example}, ground_truth_example={ground_truth_example}. "
            "Use a ground truth file generated for the same chunking run."
        )

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )

    return {
        "precision": float(round(precision, 4)),
        "recall": float(round(recall, 4)),
        "f1_score": float(round(f1, 4)),
        "sample_size": float(len(y_true)),
    }
