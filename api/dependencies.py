"""FastAPI dependency-injection providers.

Each function receives the current Request and pulls the matching
component off app.state (set once at startup by the lifespan).
Route handlers declare them with Depends(...); tests override them
via app.dependency_overrides[get_xxx] = lambda: mock.
"""
from __future__ import annotations

from fastapi import Request

from src.generation.generator import Generator
from src.ingestion.fda_client import FDAClient
from src.ingestion.rxnorm_client import RxNormClient
from src.retrieval.interaction_checker import InteractionChecker
from src.retrieval.retrieval import Retriever


def get_rxnorm(request: Request) -> RxNormClient:
    return request.app.state.rxnorm


def get_fda(request: Request) -> FDAClient:
    return request.app.state.fda


def get_interaction_checker(request: Request) -> InteractionChecker:
    return request.app.state.interaction_checker


def get_retriever(request: Request) -> Retriever:
    return request.app.state.retriever


def get_generator(request: Request) -> Generator:
    return request.app.state.generator