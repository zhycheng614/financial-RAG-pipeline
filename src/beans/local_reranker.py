"""
Local reranking service using nexaai package with multiprocessing support.

This module provides a local reranking solution that:
1. Loads the Jina reranker model locally using nexaai package
2. Uses multiprocessing to handle concurrent reranking requests
3. Each worker loads the model once at startup for efficiency
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple
from logging import getLogger
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import asyncio

from utils.math import softmax

logger = getLogger(__name__)

# Global variable to hold the reranker model in each worker process
_worker_reranker = None
_worker_model_path = None
_worker_tokenizer_path = None


def _init_worker(model_path: str, tokenizer_path: Optional[str] = None):
    """
    Initialize the worker process by loading the reranker model.
    This function is called once when each worker process starts.
    
    Args:
        model_path: Path to the GGUF model file
        tokenizer_path: Path to the tokenizer JSON file
    """
    global _worker_reranker, _worker_model_path, _worker_tokenizer_path
    
    try:
        from nexaai import ModelConfig, Reranker
        
        _worker_model_path = model_path
        _worker_tokenizer_path = tokenizer_path
        
        # Create model config - use 0 for CPU, or adjust for GPU
        model_config = ModelConfig(n_gpu_layers=0)
        
        # Load the reranker model
        # plugin_id="nexaml" for cross-platform support, model_name="jina-rerank" for Jina reranker
        _worker_reranker = Reranker(
            model_path=model_path,
            plugin_id="nexaml",
            config=model_config,
            model_name="jina-rerank",
            tokenizer_path=tokenizer_path,
        )
        
        if _worker_reranker is None:
            raise RuntimeError(f"Failed to load rerank model: {model_path}")
            
        logger.info(f"Worker process {mp.current_process().name} loaded reranker model successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize worker with reranker: {e}")
        raise


def _rerank_in_worker(
    query: str,
    documents: List[str],
    batch_size: int,
    use_softmax: bool
) -> List[float]:
    """
    Perform reranking in a worker process using the pre-loaded model.
    
    Args:
        query: The query string
        documents: List of documents to rerank
        batch_size: Batch size for processing
        use_softmax: Whether to apply softmax normalization
        
    Returns:
        List of rerank scores
    """
    global _worker_reranker
    
    if _worker_reranker is None:
        raise RuntimeError("Reranker model not initialized in worker process")
    
    if not query or not documents:
        return []
    
    try:
        if batch_size is None or batch_size >= len(documents):
            # Process all documents at once
            result = _worker_reranker.rerank(
                query,
                documents,
                batch_size=len(documents),
                normalize=use_softmax,
                normalize_method="softmax" if use_softmax else None,
            )
            return result.scores
        else:
            # Process in batches - always get raw scores first
            all_scores = []
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                result = _worker_reranker.rerank(
                    query,
                    batch,
                    batch_size=len(batch),
                    normalize=False,
                    normalize_method=None,
                )
                all_scores.extend(result.scores)
            
            # Apply softmax normalization to the complete set if needed
            if use_softmax:
                all_scores = softmax(all_scores).tolist()
            
            return all_scores
            
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error in worker rerank: {error_message}")
        raise


class BaseRerankService(ABC):
    """Abstract base class for rerank services."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[str],
        batch_size: Optional[int] = 3,
        use_softmax: bool = True,
    ) -> List[float]:
        """
        Rerank documents based on query relevance.

        Args:
            query: The query string
            documents: List of documents to rerank
            batch_size: Optional batch size for processing
            use_softmax: Whether to apply softmax normalization to scores

        Returns:
            List of rerank scores for each document
        """
        pass

    @abstractmethod
    def rerank_sort(
        self,
        query: str,
        documents: List[str],
        sort_along: List[Any],
        batch_size: Optional[int] = 3,
        top_k: Optional[int] = None,
        use_softmax: bool = False,
    ) -> Tuple[List[Any], List[float]]:
        """
        Rerank documents and sort the corresponding sort_along list based on rerank scores.

        Args:
            query: The query string
            documents: List of documents to rerank
            sort_along: List of items to sort alongside documents
            batch_size: Optional batch size for processing
            top_k: Optional parameter to return only top k results
            use_softmax: Whether to apply softmax normalization to scores

        Returns:
            Tuple of (sorted_sort_along, sorted_scores)
        """
        pass

    @abstractmethod
    async def rerank_async(
        self,
        query: str,
        documents: List[str],
        batch_size: Optional[int] = 3,
        use_softmax: bool = True,
    ) -> List[float]:
        """Asynchronous version of rerank method."""
        pass

    @abstractmethod
    async def rerank_sort_async(
        self,
        query: str,
        documents: List[str],
        sort_along: List[Any],
        batch_size: Optional[int] = 3,
        top_k: Optional[int] = None,
        use_softmax: bool = False,
    ) -> Tuple[List[Any], List[float]]:
        """Asynchronous version of rerank_sort method."""
        pass

    def close(self):
        """Clean up resources. Default implementation does nothing."""
        pass

    async def close_async(self):
        """Async cleanup. Default implementation does nothing."""
        pass


class LocalRerankService(BaseRerankService):
    """
    Local reranking service using nexaai package with multiprocessing.
    
    Uses a ProcessPoolExecutor where each worker process loads the model once
    at startup and handles reranking requests assigned to it.
    """
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        num_workers: int = 2,
    ):
        """
        Initialize the local rerank service.
        
        Args:
            model_path: Path to the GGUF model file
            tokenizer_path: Path to the tokenizer JSON file
            num_workers: Number of worker processes to spawn
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.num_workers = num_workers
        
        # Create process pool with initializer to load model in each worker
        logger.info(f"Initializing LocalRerankService with {num_workers} workers")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Tokenizer path: {tokenizer_path}")
        
        self._executor = ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
            initargs=(model_path, tokenizer_path),
        )
        
        # Pre-warm the workers by submitting a dummy task to each
        self._warm_up_workers()
        
        logger.info("LocalRerankService initialized successfully")
    
    def _warm_up_workers(self):
        """Pre-warm all workers to ensure models are loaded."""
        logger.info("Warming up worker processes...")
        futures = []
        for _ in range(self.num_workers):
            future = self._executor.submit(
                _rerank_in_worker,
                "test query",
                ["test document"],
                1,
                False
            )
            futures.append(future)
        
        # Wait for all workers to complete warm-up
        for future in futures:
            try:
                future.result(timeout=60)  # 60 second timeout for model loading
            except Exception as e:
                logger.warning(f"Worker warm-up had an issue: {e}")
        
        logger.info("All workers warmed up")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        batch_size: Optional[int] = 3,
        use_softmax: bool = True,
    ) -> List[float]:
        """
        Rerank documents using the local model via worker process.
        
        Args:
            query: The query string
            documents: List of documents to rerank
            batch_size: Batch size for processing
            use_softmax: Whether to apply softmax normalization
            
        Returns:
            List of rerank scores
        """
        if not query or not documents:
            return []
        
        # Submit task to worker pool
        future = self._executor.submit(
            _rerank_in_worker,
            query,
            documents,
            batch_size if batch_size is not None else len(documents),
            use_softmax
        )
        
        return future.result()
    
    def rerank_sort(
        self,
        query: str,
        documents: List[str],
        sort_along: List[Any],
        batch_size: Optional[int] = 3,
        top_k: Optional[int] = None,
        use_softmax: bool = False,
    ) -> Tuple[List[Any], List[float]]:
        """
        Rerank documents and sort the corresponding sort_along list.
        
        Args:
            query: The query string
            documents: List of documents to rerank
            sort_along: List of items to sort alongside documents
            batch_size: Batch size for processing
            top_k: Optional parameter to return only top k results
            use_softmax: Whether to apply softmax normalization
            
        Returns:
            Tuple of (sorted_sort_along, sorted_scores)
        """
        if len(documents) != len(sort_along):
            raise ValueError(
                f"documents and sort_along must have same length. "
                f"Got {len(documents)} documents and {len(sort_along)} sort_along items"
            )
        
        if not query or not documents:
            return [], []
        
        # Get rerank scores
        scores = self.rerank(query, documents, batch_size, use_softmax=use_softmax)
        
        # Sort by scores
        indices = list(range(len(scores)))
        sorted_indices = sorted(indices, key=lambda i: scores[i], reverse=True)
        
        sorted_scores = [scores[i] for i in sorted_indices]
        sorted_sort_along = [sort_along[i] for i in sorted_indices]
        
        # Apply top_k truncation if specified
        if top_k is not None and top_k > 0:
            sorted_scores = sorted_scores[:top_k]
            sorted_sort_along = sorted_sort_along[:top_k]
        
        return sorted_sort_along, sorted_scores
    
    async def rerank_async(
        self,
        query: str,
        documents: List[str],
        batch_size: Optional[int] = 3,
        use_softmax: bool = True,
    ) -> List[float]:
        """
        Asynchronously rerank documents using the local model.
        
        Uses run_in_executor to run the blocking operation in the process pool.
        """
        if not query or not documents:
            return []
        
        loop = asyncio.get_event_loop()
        
        # Run the blocking operation in the process pool
        result = await loop.run_in_executor(
            self._executor,
            _rerank_in_worker,
            query,
            documents,
            batch_size if batch_size is not None else len(documents),
            use_softmax
        )
        
        return result
    
    async def rerank_sort_async(
        self,
        query: str,
        documents: List[str],
        sort_along: List[Any],
        batch_size: Optional[int] = 3,
        top_k: Optional[int] = None,
        use_softmax: bool = False,
    ) -> Tuple[List[Any], List[float]]:
        """
        Asynchronously rerank documents and sort the corresponding sort_along list.
        """
        if len(documents) != len(sort_along):
            raise ValueError(
                f"documents and sort_along must have same length. "
                f"Got {len(documents)} documents and {len(sort_along)} sort_along items"
            )
        
        if not query or not documents:
            return [], []
        
        # Get rerank scores asynchronously
        scores = await self.rerank_async(query, documents, batch_size, use_softmax=use_softmax)
        
        # Sort by scores
        indices = list(range(len(scores)))
        sorted_indices = sorted(indices, key=lambda i: scores[i], reverse=True)
        
        sorted_scores = [scores[i] for i in sorted_indices]
        sorted_sort_along = [sort_along[i] for i in sorted_indices]
        
        # Apply top_k truncation if specified
        if top_k is not None and top_k > 0:
            sorted_scores = sorted_scores[:top_k]
            sorted_sort_along = sorted_sort_along[:top_k]
        
        return sorted_sort_along, sorted_scores
    
    def close(self):
        """Shutdown the process pool."""
        logger.info("Shutting down LocalRerankService worker pool")
        self._executor.shutdown(wait=True)
        logger.info("LocalRerankService shutdown complete")
    
    async def close_async(self):
        """Async shutdown of the process pool."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.close)
