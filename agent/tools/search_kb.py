"""Knowledge-base search tool using TF-IDF cosine similarity."""

import json
import logging
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import settings

logger = logging.getLogger(__name__)


def _load_documents() -> list[dict[str, Any]]:
    """Load all knowledge-base documents from the configured directory.

    Returns:
        list[dict[str, Any]]: A flat list of documents from every
            `*.json` file under `knowledge_base_dir` that contains a
            `documents` array. Returns an empty list if none are found.
    """
    documents: list[dict[str, Any]] = []
    kb_dir = Path(settings.knowledge_base_dir)
    if not kb_dir.exists():
        return documents

    for path in sorted(kb_dir.glob("*.json")):
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load knowledge base file %s: %s", path, exc)
            continue

        for doc in data.get("documents", []):
            documents.append({**doc, "source_file": path.name})

    return documents


class KnowledgeBaseSearch:
    """TF-IDF based search engine over the local knowledge base."""

    def __init__(self) -> None:
        """Load knowledge-base documents and fit the TF-IDF vectorizer."""
        self.documents = _load_documents()
        self._corpus = [doc.get("content_ar", "") for doc in self.documents]

        if self._corpus:
            self._vectorizer = TfidfVectorizer()
            self._matrix = self._vectorizer.fit_transform(self._corpus)
        else:
            self._vectorizer = None
            self._matrix = None

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Search the knowledge base for documents relevant to a query.

        Args:
            query: Free-text search query (typically the customer message
                or intent label).
            top_k: Maximum number of results to return.

        Returns:
            list[dict[str, Any]]: Up to `top_k` documents sorted by
                relevance, each augmented with a `score` field. Returns
                an empty list if the query is empty or the knowledge base
                has no documents.
        """
        if not query or not self.documents or self._vectorizer is None:
            return []

        try:
            query_vector = self._vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self._matrix)[0]
        except ValueError as exc:
            logger.warning("KB search failed for query '%s': %s", query, exc)
            return []

        ranked_indices = similarities.argsort()[::-1][:top_k]
        results = []
        for idx in ranked_indices:
            score = float(similarities[idx])
            if score <= 0:
                continue
            results.append({**self.documents[idx], "score": score})
        return results


_kb_search_instance: KnowledgeBaseSearch | None = None


def search_kb(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Search the knowledge base for documents relevant to a query.

    Lazily initializes a module-level `KnowledgeBaseSearch` instance so the
    TF-IDF index is built only once.

    Args:
        query: Free-text search query.
        top_k: Maximum number of results to return.

    Returns:
        list[dict[str, Any]]: Up to `top_k` matching documents with scores.
    """
    global _kb_search_instance
    if _kb_search_instance is None:
        _kb_search_instance = KnowledgeBaseSearch()
    return _kb_search_instance.search(query, top_k=top_k)
