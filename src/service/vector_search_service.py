from typing import List, Tuple 
from beans.embedder import Embedder
from beans.faiss_manager import FaissIndex
from logging import getLogger

logger = getLogger(__name__)

class VectorSearchService:
    def __init__(
        self,
        faiss_index: FaissIndex,
        embedder: Embedder,
        default_limit: int = 10
    ) -> None:
        """Initialize a vector search engine using FAISS.

        Args:
            faiss_index: The FAISS index to search
            embedder: The embedding generator to create query embeddings
            default_limit: Default number of results to return
        """
        self.faiss_index = faiss_index
        self.embedder = embedder
        self.default_limit = default_limit

    def search(
        self,
        query: str,
        subset_ids: List[int] | None = None,
        exclude_ids: List[int] | None = None,
        limit: int = None
    ) -> Tuple[List[int], List[float]]:
        """Perform semantic search with filtering to a subset of chunk IDs.

        Args:
            query: The search query
            subset_ids: List of chunk IDs to search within

        Returns:
            Tuple of (chunk_ids, distances)
        """
        query_vector = self.embedder.generate_embedding(query)
        
        if subset_ids is None:
            logger.warning("No subset IDs provided, searching all chunks")

        distances, ids = self.faiss_index.search(
            query_vector,
            k=limit if limit is not None else self.default_limit,
            subset_ids=subset_ids,
            exclude_ids=exclude_ids
        )

        result_ids = ids[0].tolist()
        result_distances = distances[0].tolist()
        valid_results = [(id, dist) for id, dist in zip(result_ids, result_distances) if id != -1]

        if not valid_results:
            return [], []

        chunk_ids, distances = zip(*valid_results)
        return list(chunk_ids), list(distances)
    
    async def search_async(
        self,
        query: str,
        subset_ids: List[int] | None = None,
        exclude_ids: List[int] | None = None,
        limit: int = None
    ) -> Tuple[List[int], List[float]]:
        """Perform semantic search asynchronously with filtering to a subset of chunk IDs.

        Args:
            query: The search query
            subset_ids: List of chunk IDs to search within
            exclude_ids: List of chunk IDs to exclude
            limit: Number of results to return

        Returns:
            Tuple of (chunk_ids, distances)
        """
        query_vector = await self.embedder.generate_embedding_async(query)
        
        if subset_ids is None:
            logger.warning("No subset IDs provided, searching all chunks")

        distances, ids = self.faiss_index.search(
            query_vector,
            k=limit if limit is not None else self.default_limit,
            subset_ids=subset_ids,
            exclude_ids=exclude_ids
        )

        result_ids = ids[0].tolist()
        result_distances = distances[0].tolist()
        valid_results = [(id, dist) for id, dist in zip(result_ids, result_distances) if id != -1]

        if not valid_results:
            return [], []

        chunk_ids, distances = zip(*valid_results)
        return list(chunk_ids), list(distances)