from openai import OpenAI

from app.chroma_store import ChromaStore
from app.config import settings
from app.schemas import RetrievedChunk


class Retriever:
    def __init__(self):
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.store = ChromaStore()

    def retrieve(self, question: str, k: int) -> list[RetrievedChunk]:
        response = self.client.embeddings.create(
            model=settings.openai_embedding_model,
            input=question,
        )
        query_embedding = response.data[0].embedding

        result = self.store.search(query_embedding, k)

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        ids = result.get("ids", [[]])[0]

        chunks = []
        for i, text in enumerate(documents):
            chunks.append(
                RetrievedChunk(
                    chunk_id=ids[i],
                    text=text,
                    metadata=metadatas[i] or {},
                    retrieval_distance=distances[i] if i < len(distances) else None,
                )
            )
        return chunks
