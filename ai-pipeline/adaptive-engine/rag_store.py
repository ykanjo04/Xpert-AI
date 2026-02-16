from __future__ import annotations

from typing import List


class RAGStore:
    def __init__(self, persist_dir: str):
        try:
            import chromadb
            from chromadb.utils import embedding_functions
        except ImportError as exc:
            raise ImportError(
                "chromadb is required for RAGStore. Install with `pip install chromadb`."
            ) from exc

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self._collection = self._client.get_or_create_collection(
            name="medical_textbook",
            embedding_function=self._embedding_fn,
        )

    def add_documents(self, texts: List[str], ids: List[str]) -> None:
        if not texts or not ids:
            return
        self._collection.add(documents=texts, ids=ids)

    def query(self, query_text: str, k: int = 4) -> List[str]:
        if not query_text:
            return []
        results = self._collection.query(query_texts=[query_text], n_results=k)
        documents = results.get("documents", [[]])[0]
        return documents or []
