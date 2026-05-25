"""Cross-cutting helpers: LLM client, embedding model, logging, seeding."""

from scaffold.utils.llm import LLMClient, ChatMessage, OpenAIChatClient, MockChatClient
from scaffold.utils.embedding import (
    Embedder,
    SentenceTransformerEmbedder,
    HashEmbedder,
)
from scaffold.utils.logging import get_logger, configure_logging
from scaffold.utils.seed import set_seed

__all__ = [
    "LLMClient",
    "ChatMessage",
    "OpenAIChatClient",
    "MockChatClient",
    "Embedder",
    "SentenceTransformerEmbedder",
    "HashEmbedder",
    "get_logger",
    "configure_logging",
    "set_seed",
]
