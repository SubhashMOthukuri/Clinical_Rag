"""Cusotome Exceptions for the Pinecone vectore Database"""

class PineconeStoreError(Exception):
    """ Exception occure in the Pinecone vectore database"""

class PineconeUnavailable(PineconeStoreError):
    """Pinecone down or circuite breaker open -> caller falls back to FDA only."""

class PineconeTimeout(PineconeStoreError):
    """Call exceeded latency budget -> caller fallbs back to FDA only."""

class PineconeIndexNotFound(PineconeStoreError):
    """Index doesn't exist on startup -> fatal, app should refuse to start."""

class PineconeInvalidInput(PineconeStoreError):
    """Input are invalid"""

class PineconeRateLimited(PineconeStoreError):
    """ Pinecone returned 429 -> caller retries with backoff. Not a service outage."""
    def __init__(self, retry_after_s : float):
        self._retry_after_s = retry_after_s
        super().__init__(f"Rate limited, retry after {retry_after_s}s")
                                                                                                                                                                                                                                                                                                                                                                                                                       