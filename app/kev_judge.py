from typing import Any, Dict
import httpx

from app.config import settings
from app.schemas import JudgedChunk, RetrievedChunk


class KevJudge:
    def __init__(self):
        self.base_url = settings.kev_api_url.rstrip("/")

    def _build_state(self, question: str, chunk: RetrievedChunk) -> str:
        return (
            f"USER QUESTION:\n{question}\n\n"
            f"RETRIEVED CONSTITUTION PASSAGE:\n{chunk.text}"
        )

    def judge(self, question: str, chunks: list[RetrievedChunk]) -> list[JudgedChunk]:
        if not chunks:
            return []

        questions: Dict[str, Dict[str, Any]] = {}
        for index, _ in enumerate(chunks):
            questions[f"chunk_{index}"] = {
                "type": "score",
                "instructions": (
                    "How relevant is this retrieved Constitution of India passage "
                    "for answering the user's question? Judge only the passage's "
                    "usefulness as evidence for answering the question."
                ),
                "criteria": [
                    "Not relevant",
                    "Somewhat relevant",
                    "Relevant",
                    "Highly relevant",
                ],
            }

        # KEV evaluates multiple questions against one shared state. To keep
        # each chunk independent, we make one KEV request per chunk here.
        judged = []
        with httpx.Client(timeout=60.0) as client:
            for index, chunk in enumerate(chunks):
                payload = {
                    "state": self._build_state(question, chunk),
                    "model": settings.kev_model,
                    "questions": {
                        f"chunk_{index}": questions[f"chunk_{index}"]
                    },
                }

                response = client.post(
                    f"{self.base_url}/v1/systemone",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                answer = data["answers"][f"chunk_{index}"]
                judged.append(
                    JudgedChunk(
                        **chunk.model_dump(),
                        kev_score=float(answer.get("score", 0.0)),
                        kev_confidence=answer.get("confidence"),
                        kev_probabilities=answer.get("probabilities", {}),
                    )
                )

        return sorted(
            judged,
            key=lambda item: item.kev_score,
            reverse=True,
        )
