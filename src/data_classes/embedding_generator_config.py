from dataclasses import dataclass
from constants import EMBEDDING_MAX_BATCH_SIZE, DEFAULT_EMBEDDING_DIM, DEFAULT_EMBEDDING_MODEL

@dataclass
class EmbedderConfig:
    max_batch_size: int = EMBEDDING_MAX_BATCH_SIZE
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    model_name: str = DEFAULT_EMBEDDING_MODEL