from pathlib import Path
import re
import pandas as pd


def load_plan_csv(path: str | Path, text_column: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    if text_column not in data.columns:
        raise ValueError(f"Missing required column: {text_column}")
    data[text_column] = data[text_column].fillna("").astype(str)
    return data


def _resolve_text_column(df: pd.DataFrame, candidate_columns: list[str]) -> str:
    for column in candidate_columns:
        if column in df.columns:
            return column
    raise ValueError(f"Missing text column. Expected one of: {candidate_columns}")


def _chunk_text_by_paragraph(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    parts = re.split(r"\n\s*\n+", text.strip())
    cleaned = [part.strip() for part in parts if part and part.strip()]
    return cleaned or [text.strip()]


def _resolve_section_name(row: pd.Series, fallback: str = "General") -> str:
    section_candidates = ["section_name", "section", "pillar", "theme"]
    for col in section_candidates:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col]).strip()
    return fallback


def build_chunk_dataframe(
    strategic_csv_path: str | Path,
    action_csv_path: str | Path,
) -> pd.DataFrame:
    strategic_df = pd.read_csv(strategic_csv_path)
    action_df = pd.read_csv(action_csv_path)
    return build_chunk_dataframe_from_dataframes(strategic_df, action_df)


def build_chunk_dataframe_from_dataframes(
    strategic_df: pd.DataFrame,
    action_df: pd.DataFrame,
) -> pd.DataFrame:

    strategic_text_col = _resolve_text_column(strategic_df, ["goal_text", "text", "description"])
    action_text_col = _resolve_text_column(action_df, ["action_text", "text", "description"])

    chunk_rows: list[dict[str, str]] = []

    for _, row in strategic_df.iterrows():
        source_id = str(row.get("goal_id", "")).strip() or f"strategic_{len(chunk_rows) + 1}"
        section_name = _resolve_section_name(row)
        chunks = _chunk_text_by_paragraph(str(row.get(strategic_text_col, "")))
        for chunk_index, chunk in enumerate(chunks, start=1):
            chunk_rows.append(
                {
                    "chunk_id": f"strategic::{source_id}::{chunk_index}",
                    "text": chunk,
                    "document_type": "strategic",
                    "section_name": section_name,
                }
            )

    for _, row in action_df.iterrows():
        source_id = str(row.get("action_id", "")).strip() or f"action_{len(chunk_rows) + 1}"
        section_name = _resolve_section_name(row)
        chunks = _chunk_text_by_paragraph(str(row.get(action_text_col, "")))
        for chunk_index, chunk in enumerate(chunks, start=1):
            chunk_rows.append(
                {
                    "chunk_id": f"action::{source_id}::{chunk_index}",
                    "text": chunk,
                    "document_type": "action",
                    "section_name": section_name,
                }
            )

    chunks_df = pd.DataFrame(chunk_rows)
    if chunks_df.empty:
        return pd.DataFrame(columns=["chunk_id", "text", "document_type", "section_name"])

    chunks_df["text"] = chunks_df["text"].fillna("").astype(str)
    chunks_df["document_type"] = chunks_df["document_type"].fillna("").astype(str)
    chunks_df["section_name"] = chunks_df["section_name"].fillna("General").astype(str)
    chunks_df = chunks_df[chunks_df["text"].str.strip() != ""].reset_index(drop=True)
    return chunks_df
