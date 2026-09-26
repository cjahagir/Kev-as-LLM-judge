import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    openai_embedding_model: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )
    kev_api_url: str = os.getenv("KEV_API_URL", "http://127.0.0.1:8009")
    kev_model: str = os.getenv("KEV_MODEL", "kev-latest")
    chroma_path: str = os.getenv("CHROMA_PATH", "./chroma_db")
    chroma_collection: str = os.getenv(
        "CHROMA_COLLECTION", "constitution_of_india"
    )
    default_retrieve_k: int = int(os.getenv("RETRIEVE_K", "8"))
    default_rerank_k: int = int(os.getenv("RERANK_K", "4"))

settings = Settings()
