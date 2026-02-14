from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.data_loader import build_chunk_dataframe
from src.embedding_engine import EmbeddingEngine
from src.vector_store import VectorStore


def load_and_chunk_plans(
    strategic_csv_path: str | Path = "data/input/strategic_plan.csv",
    action_csv_path: str | Path = "data/input/action_plan.csv",
) -> pd.DataFrame:
    chunks_df = build_chunk_dataframe(strategic_csv_path, action_csv_path)
    required_columns = {"chunk_id", "text", "document_type", "section_name"}
    missing = required_columns.difference(chunks_df.columns)
    if missing:
        raise ValueError(f"Chunk dataframe missing required columns: {sorted(missing)}")
    return chunks_df[["chunk_id", "text", "document_type", "section_name"]].copy()


def embed_and_store_chunks(chunks_df: pd.DataFrame, persist_directory: str = "./chroma_db") -> int:
    if chunks_df.empty:
        return 0

    embedder = EmbeddingEngine()
    vector_store = VectorStore(persist_directory=persist_directory)

    ids = chunks_df["chunk_id"].astype(str).tolist()
    texts = chunks_df["text"].astype(str).tolist()

    metadatas = [
        {
            "document_type": row.document_type,
            "section_name": row.section_name,
        }
        for row in chunks_df.itertuples(index=False)
    ]

    embeddings = embedder.embed_texts(texts)
    vector_store.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return len(ids)


def run_ingestion(
    strategic_csv_path: str | Path = "data/input/strategic_plan.csv",
    action_csv_path: str | Path = "data/input/action_plan.csv",
    persist_directory: str = "./chroma_db",
) -> pd.DataFrame:
    chunks_df = load_and_chunk_plans(strategic_csv_path, action_csv_path)
    embed_and_store_chunks(chunks_df, persist_directory=persist_directory)
    return chunks_df


if __name__ == "__main__":
    dataframe = run_ingestion()
    print(f"Ingestion complete. Stored {len(dataframe)} chunks in ChromaDB.")
