from openai import OpenAI

from app.config import settings
from app.schemas import JudgedChunk


class AnswerGenerator:
    def __init__(self):
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        self.client = OpenAI(api_key=settings.openai_api_key)

    def generate(self, question: str, chunks: list[JudgedChunk]) -> str:
        context_parts = []
        for index, chunk in enumerate(chunks, start=1):
            page = chunk.metadata.get("page", "unknown")
            context_parts.append(
                f"[SOURCE {index} | page {page}]\n{chunk.text}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""
You are answering questions using the Constitution of India as the knowledge base.

User question:
{question}

Use ONLY the supplied Constitution passages as evidence.
If the passages do not contain enough information to answer, say that the
retrieved material is insufficient. Do not invent constitutional provisions,
articles, sections, cases, or facts.

Retrieved passages:
{context}
"""

        response = self.client.chat.completions.create(
            model=settings.openai_chat_model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer accurately and concisely from the provided "
                        "constitutional evidence."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )

        return response.choices[0].message.content or ""
