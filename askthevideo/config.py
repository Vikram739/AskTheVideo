import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# Resolve .env from the project root so the app works no matter which directory it is launched from.
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


class MissingApiKeyError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    groq_model: str


def load_settings() -> Settings:
    load_dotenv(ENV_PATH)
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise MissingApiKeyError("GROQ_API_KEY is not set.")
    model = os.getenv("GROQ_MODEL", "").strip() or DEFAULT_GROQ_MODEL
    return Settings(groq_api_key=api_key, groq_model=model)
