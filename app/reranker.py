from app.kev_judge import KevJudge
from app.schemas import JudgedChunk, RetrievedChunk


class KevReranker:
    def __init__(self):
        self.judge = KevJudge()

    def rerank(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[JudgedChunk]:
        if not chunks:
            return []

        judged = self.judge.judge(question, chunks)
        return judged[:top_k]
