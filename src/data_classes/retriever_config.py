from dataclasses import dataclass
from constants import (
    DEFAULT_SEMANTIC_FILTERING_THRESHOLD,
    DEFAULT_FINAL_CHUNKS_TO_ASSEMBLE_LIMIT,
    DEFAULT_FTS_RETRIEVAL_LIMIT,
    DEFAULT_SEMANTIC_RETRIEVAL_LIMIT,
    DEFAULT_RRF_K,
)

@dataclass
class RetrieverConfig:
    semantic_filtering_threshold: float = DEFAULT_SEMANTIC_FILTERING_THRESHOLD
    final_chunks_to_assemble_limit: int = DEFAULT_FINAL_CHUNKS_TO_ASSEMBLE_LIMIT
    fts_retrieval_limit: int = DEFAULT_FTS_RETRIEVAL_LIMIT
    semantic_retrieval_limit: int = DEFAULT_SEMANTIC_RETRIEVAL_LIMIT
    reciprocal_rank_fusion_k: int = DEFAULT_RRF_K