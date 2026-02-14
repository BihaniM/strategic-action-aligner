from typing import Any
import chromadb
from src.config import CHROMA_COLLECTION


class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db") -> None:
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(CHROMA_COLLECTION)

    def upsert(self, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict[str, Any]]) -> None:
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(self, query_embeddings: list[list[float]], n_results: int = 1) -> dict[str, Any]:
        return self.collection.query(query_embeddings=query_embeddings, n_results=n_results)

    def upsert_chunk_records(self, records: list[dict[str, Any]], embeddings: list[list[float]]) -> None:
        ids = [str(record["chunk_id"]) for record in records]
        documents = [str(record["text"]) for record in records]
        metadatas = [
            {
                "document_type": str(record.get("document_type", "")),
                "section_name": str(record.get("section_name", "General")),
            }
            for record in records
        ]
        self.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
