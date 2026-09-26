from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader
from openai import OpenAI

from app.chroma_store import ChromaStore
from app.config import settings


def extract_pages(pdf_path: str):
    reader = PdfReader(pdf_path)
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        if text:
            yield page_number, text


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        if end >= len(text):
            break
        start = end - overlap


def build_chunks(pdf_path: str):
    records = []
    for page_number, page_text in extract_pages(pdf_path):
        for chunk_index, chunk in enumerate(chunk_text(page_text)):
            records.append(
                {
                    "id": f"page-{page_number}-chunk-{chunk_index}",
                    "text": chunk,
                    "metadata": {
                        "page": page_number,
                        "chunk_index": chunk_index,
                        "source": Path(pdf_path).name,
                    },
                }
            )
    return records


def ingest(pdf_path: str):
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    client = OpenAI(api_key=settings.openai_api_key)
    store = ChromaStore()
    records = build_chunks(pdf_path)

    batch_size = 100
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        response = client.embeddings.create(
            model=settings.openai_embedding_model,
            input=[item["text"] for item in batch],
        )
        embeddings = [item.embedding for item in response.data]

        store.upsert(
            ids=[item["id"] for item in batch],
            documents=[item["text"] for item in batch],
            embeddings=embeddings,
            metadatas=[item["metadata"] for item in batch],
        )

    return len(records)
