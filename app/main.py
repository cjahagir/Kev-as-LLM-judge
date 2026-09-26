from fastapi import FastAPI, HTTPException

from app.config import settings
from app.generator import AnswerGenerator
from app.reranker import KevReranker
from app.retriever import Retriever
from app.schemas import AskRequest, AskResponse

app = FastAPI(
    title="Constitution RAG with KEV Judge",
    description=(
        "Backend-only RAG system using ChromaDB for retrieval, "
        "KEV for relevance judging/reranking, and OpenAI for answer generation."
    ),
    version="0.1.0",
)

retriever = Retriever()
reranker = KevReranker()
generator = AnswerGenerator()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "kev_api_url": settings.kev_api_url,
        "chroma_collection": settings.chroma_collection,
        "indexed_chunks": retriever.store.count,
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if request.rerank_k > request.retrieve_k:
        raise HTTPException(
            status_code=400,
            detail="rerank_k cannot be greater than retrieve_k",
        )

    try:
        retrieved = retriever.retrieve(
            request.question,
            request.retrieve_k,
        )

        if not retrieved:
            raise HTTPException(
                status_code=404,
                detail="No indexed Constitution passages were found.",
            )

        reranked = reranker.rerank(
            request.question,
            retrieved,
            request.rerank_k,
        )

        answer = generator.generate(
            request.question,
            reranked,
        )

        return AskResponse(
            question=request.question,
            answer=answer,
            retrieved_chunks=retrieved,
            reranked_chunks=reranked,
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )
