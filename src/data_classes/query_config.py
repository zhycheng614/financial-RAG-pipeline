from dataclasses import dataclass
from data_classes.retriever_config import RetrieverConfig

@dataclass
class QueryConfig:
    retriever_config: RetrieverConfig
    rerank_keep_threshold: float    # scores below this threshold will be discarded
    rerank_cliff_cutoff_score_difference: float  # if score drops by more than this from first score, cutoff at that point
    
    run_rewrite: bool = True
    run_retrieve: bool = True
    run_rerank: bool = True
    run_with_keyword_extraction: bool = True
    
    max_item_to_rerank: int = None   # the main concern of reranking is latency. This config should be tuned along with chunk size.
                                     # empirically, a total of 20000 characters take 2 seconds to rerank. unlimited if not specified.