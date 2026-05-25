"""Custom exceptions for retriever module.

Retriever's normal failure mode is returning an empty list (caller falls back
to FDA-only). These exceptions are for cases the caller must handle explicitly.
"""


class RetrievalError(Exception):
    """Base for all retrieval failures."""


class RerankerUnavailable(RetrievalError):
    """Cross-encoder model failed to load or score → skip reranking, use raw scores."""