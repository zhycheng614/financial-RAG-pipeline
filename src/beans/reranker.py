from typing import List, Tuple, Any, Optional
from constants import JINA_API_KEY, DEFAULT_RERANKER_MODEL, DEFAULT_RERANKER_URL
from logging import getLogger
from utils.math import softmax
import asyncio
import random
import requests
import httpx

logger = getLogger(__name__)

# Number of times to retry a Jina rerank API call on transient failures
# (HTTP 429, 5xx, or malformed responses missing the `results` key).
_JINA_MAX_RETRIES = 5
_JINA_BACKOFF_BASE_S = 1.0



class JinaRerankService():
    def __init__(self):
        self.model = DEFAULT_RERANKER_MODEL
        self.base_url = DEFAULT_RERANKER_URL
        self._async_client = None  # Lazy initialization
        
    def rerank(self, query: str, documents: List[str], batch_size: Optional[int] = 3, use_softmax: bool = True) -> List[float]:
        if not query or not documents:
            return []
        
        key = JINA_API_KEY
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {key}'
        }

        data = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "return_documents": False
        }
        try:
            response = requests.post(self.base_url, headers=headers, json=data)
            results = response.json()
            scores = [item["relevance_score"] for item in results["results"]]
            
            if use_softmax:
                scores = softmax(scores).tolist()
                
            return scores
        except Exception as e:
            error_message = str(e)
            raise type(e)(error_message) from e

    def rerank_sort(self, query: str, documents: List[str], sort_along: List[Any], 
                   batch_size: Optional[int] = 3, top_k: Optional[int] = None, use_softmax: bool = False) -> Tuple[List[Any], List[float]]:
        """
        Rerank documents and sort the corresponding sort_along list based on rerank scores.
        
        Args:
            query: The query string
            documents: List of documents to rerank
            sort_along: List of items to sort alongside documents (must be same length as documents)
            batch_size: Optional batch size for processing
            top_k: Optional parameter to return only top k results
            
        Returns:
            Tuple of (sorted_sort_along, sorted_scores)
            
        Raises:
            ValueError: If documents and sort_along have different lengths
        """
        if len(documents) != len(sort_along):
            raise ValueError(f"documents and sort_along must have same length. "
                           f"Got {len(documents)} documents and {len(sort_along)} sort_along items")
        
        if not query or not documents:
            return [], []
        
        # Get rerank scores
        scores = self.rerank(query, documents, batch_size, use_softmax=use_softmax)
        
        # Use Python list sorting
        indices = list(range(len(scores)))
        sorted_indices = sorted(indices, key=lambda i: scores[i], reverse=True)
        
        # Sort scores and sort_along based on indices
        sorted_scores = [scores[i] for i in sorted_indices]
        sorted_sort_along = [sort_along[i] for i in sorted_indices]
        
        # Apply top_k truncation if specified
        if top_k is not None and top_k > 0:
            sorted_scores = sorted_scores[:top_k]
            sorted_sort_along = sorted_sort_along[:top_k]
        
        return sorted_sort_along, sorted_scores

    async def rerank_async(self, query: str, documents: List[str], batch_size: Optional[int] = 3, use_softmax: bool = True) -> List[float]:
        """Asynchronous version of rerank method."""
        if not query or not documents:
            return []
        
        # Lazy initialization of async client
        if self._async_client is None:
            self._async_client = httpx.AsyncClient()
        
        key = JINA_API_KEY
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {key}'
        }

        data = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "return_documents": False
        }

        last_error: Optional[Exception] = None
        for attempt in range(_JINA_MAX_RETRIES):
            try:
                response = await self._async_client.post(
                    self.base_url, headers=headers, json=data, timeout=60.0
                )
                # Treat 429/5xx as transient
                if response.status_code == 429 or response.status_code >= 500:
                    raise RuntimeError(
                        f"jina rerank HTTP {response.status_code}: {response.text[:200]}"
                    )
                results = response.json()
                if "results" not in results:
                    raise RuntimeError(
                        f"jina rerank malformed response (no `results` key): "
                        f"{str(results)[:200]}"
                    )
                scores = [item["relevance_score"] for item in results["results"]]
                if use_softmax:
                    scores = softmax(scores).tolist()
                return scores
            except Exception as e:
                last_error = e
                if attempt + 1 >= _JINA_MAX_RETRIES:
                    break
                backoff = _JINA_BACKOFF_BASE_S * (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning(
                    f"jina rerank attempt {attempt + 1}/{_JINA_MAX_RETRIES} failed: "
                    f"{e}; retrying in {backoff:.1f}s"
                )
                await asyncio.sleep(backoff)
        # All retries exhausted
        raise type(last_error)(str(last_error)) from last_error

    async def rerank_sort_async(self, query: str, documents: List[str], sort_along: List[Any],
                   batch_size: Optional[int] = 3, top_k: Optional[int] = None, use_softmax: bool = False) -> Tuple[List[Any], List[float]]:
        """
        Asynchronously rerank documents and sort the corresponding sort_along list based on rerank scores.
        
        Args:
            query: The query string
            documents: List of documents to rerank
            sort_along: List of items to sort alongside documents (must be same length as documents)
            batch_size: Optional batch size for processing
            top_k: Optional parameter to return only top k results
            
        Returns:
            Tuple of (sorted_sort_along, sorted_scores)
            
        Raises:
            ValueError: If documents and sort_along have different lengths
        """
        if len(documents) != len(sort_along):
            raise ValueError(f"documents and sort_along must have same length. "
                           f"Got {len(documents)} documents and {len(sort_along)} sort_along items")
        
        if not query or not documents:
            return [], []
        
        # Get rerank scores
        scores = await self.rerank_async(query, documents, batch_size, use_softmax=use_softmax)
        
        # Use Python list sorting
        indices = list(range(len(scores)))
        sorted_indices = sorted(indices, key=lambda i: scores[i], reverse=True)
        
        # Sort scores and sort_along based on indices
        sorted_scores = [scores[i] for i in sorted_indices]
        sorted_sort_along = [sort_along[i] for i in sorted_indices]
        
        # Apply top_k truncation if specified
        if top_k is not None and top_k > 0:
            sorted_scores = sorted_scores[:top_k]
            sorted_sort_along = sorted_sort_along[:top_k]
        
        return sorted_sort_along, sorted_scores

    async def close_async(self):
        """Clean up async resources."""
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None
    
    def close(self):
        """Clean up resources."""
        pass