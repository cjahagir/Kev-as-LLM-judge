from typing import Any, Dict, List
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    retrieve_k: int = Field(default=8, ge=1, le=50)
    rerank_k: int = Field(default=4, ge=1, le=20)


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: Dict[str, Any] = {}
    retrieval_distance: float | None = None


class JudgedChunk(RetrievedChunk):
    kev_score: float
    kev_confidence: float | None = None
    kev_probabilities: Dict[str, float] = {}


class AskResponse(BaseModel):
    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    reranked_chunks: List[JudgedChunk]
