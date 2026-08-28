from typing import List, Optional, Dict, Tuple, Union, TYPE_CHECKING
import datetime
from logging import getLogger

from beans.faiss_manager import FaissIndex
from service.vector_search_service import VectorSearchService
from beans.embedder import Embedder
from beans.reranker import JinaRerankService
from beans.local_reranker import LocalRerankService
from service.rewriting_service import RewritingService
from dao.chunk_dao import ChunkDao
from dao.document_dao import DocumentDao
from data_classes.query_config import QueryConfig
from data_classes.retriever_config import RetrieverConfig
from beans.inference_facade import InferenceFacade
from data_classes.chunk_info import ChunkInfo
from data_classes.chat_context_unit import ChatContextUnit
from data_classes.retrieval_query_unit import RetrievalQueryUnit
from utils.rrf import rrf_with_scores
from constants import DEFAULT_CHUNK_SIZE

# Type alias for reranker services (both cloud and local)
RerankService = Union[JinaRerankService, LocalRerankService]


logger = getLogger(__name__)

class QueryProcessor():

    def __init__(self, 
                 config: QueryConfig, 
                 chunk_dao: ChunkDao,
                 document_dao: DocumentDao,
                 faiss_index: FaissIndex,
                 embedder: Embedder, 
                 reranker: Optional[RerankService] = None, 
                 inference_service: Optional[InferenceFacade] = None, 
                 rewriting_service: Optional[RewritingService] = None,
                 vector_search_service: Optional[VectorSearchService] = None
                ):

        self.config = config
        self.retriever_config: RetrieverConfig = self.config.retriever_config
        if self.config.run_rerank and not reranker:
            raise ValueError("Reranking is enabled but no reranker was provided.")
        
        if self.config.run_rewrite and not inference_service:
            raise ValueError("Rewrite is enabled but no inference_service was provided.")

        # Note: we use the same default limit for now, but we can change it later
        self.vector_search_service = vector_search_service

        # Database related dependencies
        self.chunk_dao = chunk_dao
        self.document_dao = document_dao
        self.faiss_index = faiss_index
        
        self.reranker = reranker
        self.inference_service = inference_service
        self.rewriting_service = rewriting_service
        

    def process_query(self, original_query: str, document_ids: Optional[List[int]] = None) -> List[ChatContextUnit]:
        """Process a query and return chat context units with retrieved chunks.
        
        Args:
            original_query (str): The user's original query
            document_ids: Optional list of document IDs to restrict retrieval to.
                         If provided, only chunks from these documents will be searched.
                         If None, searches all documents.
            
        Returns:
            List[ChatContextUnit]: List of chat context units containing retrieved and ranked chunks
        """ 
        logger.debug(f"Starting query processing for original_query: {original_query}")

        # Step 1: Rewrite query if enabled, otherwise use original query
        if self.config.run_rewrite and self.rewriting_service:
            rewrite_service_output = self.rewriting_service.rewrite(original_query)
            logger.info(f"Query rewrite results - Original: '{original_query}' -> Clarified: '{rewrite_service_output.clarified_query}', Keywords: {rewrite_service_output.keywords}")
            query_units = [RetrievalQueryUnit(
                query=rewrite_service_output.clarified_query, 
                keywords=rewrite_service_output.keywords
            )]
        else:
            logger.info(f"Query rewrite disabled, using original query: '{original_query}'")
            # Create a simple query unit from the original query
            query_units = [RetrievalQueryUnit(query=original_query, keywords=None)]
        
        # Step 2: Retrieve chunks (optionally scoped to specific documents)
        if self.config.run_retrieve:
            chunk_ids_list = self.retrieve_chunk_ids_by_hybrid_search_multi_query(
                query_units, 
                document_ids=document_ids,
                use_keyword_extraction=self.config.run_rewrite and self.config.run_with_keyword_extraction
            )
        else:
            chunk_ids_list = []
            
        # Step 3: Convert chunk IDs to chat context units with metadata
        chat_context_units_list = self._get_chat_context_units_list_from_chunk_ids_list_preserve_order(chunk_ids_list)

        # Step 4: Rerank if enabled
        if self.config.run_rerank:
            chat_context_units_list = self._rerank(query_units, chat_context_units_list)
                
        # Step 5: Consolidate results from multiple query units
        chat_context_units = self.consolidate_results(chat_context_units_list)
        
        # Step 6: Apply final_chunks_to_assemble_limit when reranking is disabled
        # (Reranking applies its own cutoff logic, so this is only needed when reranking is off)
        if not self.config.run_rerank:
            limit = self.retriever_config.final_chunks_to_assemble_limit
            if len(chat_context_units) > limit:
                logger.info(f"Reranking disabled: truncating {len(chat_context_units)} chunks to top {limit}")
                chat_context_units = chat_context_units[:limit]

        logger.info(f"Query processing complete. Retrieved {len(chat_context_units)} chat context units")
        return chat_context_units
    
    async def process_query_async(self, original_query: str, document_ids: Optional[List[int]] = None) -> List[ChatContextUnit]:
        """Asynchronously process a query and return chat context units with retrieved chunks.
        
        Args:
            original_query (str): The user's original query
            document_ids: Optional list of document IDs to restrict retrieval to.
                         If provided, only chunks from these documents will be searched.
                         If None, searches all documents.
            
        Returns:
            List[ChatContextUnit]: List of chat context units containing retrieved and ranked chunks
        """ 
        logger.debug(f"Starting async query processing for original_query: {original_query}")

        # Step 1: Rewrite query if enabled, otherwise use original query
        if self.config.run_rewrite and self.rewriting_service:
            rewrite_service_output = await self.rewriting_service.rewrite_async(original_query)
            logger.info(f"Query rewrite results - Original: '{original_query}' -> Clarified: '{rewrite_service_output.clarified_query}', Keywords: {rewrite_service_output.keywords}")
            query_units = [RetrievalQueryUnit(
                query=rewrite_service_output.clarified_query, 
                keywords=rewrite_service_output.keywords
            )]
        else:
            logger.info(f"Query rewrite disabled, using original query: '{original_query}'")
            # Create a simple query unit from the original query
            query_units = [RetrievalQueryUnit(query=original_query, keywords=None)]
        
        # Step 2: Retrieve chunks (optionally scoped to specific documents)
        if self.config.run_retrieve:
            chunk_ids_list = await self.retrieve_chunk_ids_by_hybrid_search_multi_query_async(
                query_units, 
                document_ids=document_ids,
                use_keyword_extraction=self.config.run_rewrite and self.config.run_with_keyword_extraction
            )
        else:
            chunk_ids_list = []
            
        # Step 3: Convert chunk IDs to chat context units with metadata
        chat_context_units_list = self._get_chat_context_units_list_from_chunk_ids_list_preserve_order(chunk_ids_list)

        # Step 4: Rerank if enabled
        if self.config.run_rerank:
            chat_context_units_list = await self._rerank_async(query_units, chat_context_units_list)
                
        # Step 5: Consolidate results from multiple query units
        chat_context_units = self.consolidate_results(chat_context_units_list)
        
        # Step 6: Apply final_chunks_to_assemble_limit when reranking is disabled
        # (Reranking applies its own cutoff logic, so this is only needed when reranking is off)
        if not self.config.run_rerank:
            limit = self.retriever_config.final_chunks_to_assemble_limit
            if len(chat_context_units) > limit:
                logger.info(f"Reranking disabled: truncating {len(chat_context_units)} chunks to top {limit}")
                chat_context_units = chat_context_units[:limit]

        logger.info(f"Async query processing complete. Retrieved {len(chat_context_units)} chat context units")
        return chat_context_units

    def retrieve_chunk_ids_by_keyword_search(self, keyword: str, subset_ids: List[int] = None) -> List[int]:
        logger.debug(f"retrieve_chunk_ids_by_keyword_search: keyword: {keyword}, subset_ids: {subset_ids}")
        chunk_ids, _ = self.chunk_dao.search(keyword, subset_ids=subset_ids, limit=self.retriever_config.fts_retrieval_limit)
        logger.info(f"retrieve_chunk_ids_by_keyword_search retrieved ids: {chunk_ids}")
        return chunk_ids

    def retrieve_chunk_ids_by_semantic_search(self, query: str, subset_ids: List[int] = None) -> List[int]:
        logger.debug(f"retrieve_chunk_ids_by_semantic_search: query: {query}, subset_ids: {subset_ids}")
        ids, distances = self.vector_search_service.search(query, subset_ids=subset_ids)
        
        original_len = len(ids)
        # filter out ids with distance greater than the threshold
        logger.debug(f"id and distance before filtering: {ids}, {distances}")
        ids = [id for id, distance in zip(ids, distances) if distance <= self.retriever_config.semantic_filtering_threshold]
        filtered_len = len(ids)
        logger.debug(f"retrieve_chunk_ids_by_semantic_search filtered {original_len - filtered_len} ids")
        
        logger.info(f"retrieve_chunk_ids_by_semantic_search retrieved ids: {ids}")
        return ids
    
    async def retrieve_chunk_ids_by_semantic_search_async(self, query: str, subset_ids: List[int] = None) -> List[int]:
        logger.debug(f"retrieve_chunk_ids_by_semantic_search_async: query: {query}, subset_ids: {subset_ids}")
        ids, distances = await self.vector_search_service.search_async(query, subset_ids=subset_ids)
        
        original_len = len(ids)
        # filter out ids with distance greater than the threshold
        logger.debug(f"id and distance before filtering: {ids}, {distances}")
        ids = [id for id, distance in zip(ids, distances) if distance <= self.retriever_config.semantic_filtering_threshold]
        filtered_len = len(ids)
        logger.debug(f"retrieve_chunk_ids_by_semantic_search_async filtered {original_len - filtered_len} ids")
        
        logger.info(f"retrieve_chunk_ids_by_semantic_search_async retrieved ids: {ids}")
        return ids

    def retrieve_chunk_ids_by_hybrid_search_multi_query(self,
                                                        query_units: List[RetrievalQueryUnit],
                                                        document_ids: List[int] = None,
                                                        subset_ids: List[int] = None,
                                                        use_keyword_extraction: bool = False) -> List[List[int]]:
        """Retrieve chunk ids by hybrid search for multiple queries.

        Args:
            query_units: List of query units to search for
            document_ids: Optional list of document IDs to restrict search to. If provided, only chunks 
                         from these documents will be searched. If None, searches all documents.
            subset_ids: Optional list of chunk IDs to restrict search to
            use_keyword_extraction: Whether to use extracted keywords for FTS search

        Returns:
            List of lists of chunk ids. Each sub-list corresponds to one query unit and is 
            deduplicated by RRF, but the overall list is not deduplicated across queries.
        """
        logger.debug(f"retrieve_chunk_ids_by_hybrid_search_multi_query: "
                     f"queries: {query_units}, document_ids: {document_ids}, subset_ids: {subset_ids}")
        
        # If document_ids filter is provided, convert to chunk_ids filter
        if document_ids:
            subset_ids = self.chunk_dao.get_chunk_ids_by_document_ids(document_ids)
            logger.debug(f"Filtered to {len(subset_ids)} chunks from {len(document_ids)} documents")
        
        return [self.retrieve_chunk_ids_by_hybrid_search(query_unit, subset_ids, use_keyword_extraction=use_keyword_extraction) 
                for query_unit in query_units]
    
    async def retrieve_chunk_ids_by_hybrid_search_multi_query_async(self,
                                                        query_units: List[RetrievalQueryUnit],
                                                        document_ids: List[int] = None,
                                                        subset_ids: List[int] = None,
                                                        use_keyword_extraction: bool = False) -> List[List[int]]:
        """Asynchronously retrieve chunk ids by hybrid search for multiple queries.

        Args:
            query_units: List of query units to search for
            document_ids: Optional list of document IDs to restrict search to. If provided, only chunks 
                         from these documents will be searched. If None, searches all documents.
            subset_ids: Optional list of chunk IDs to restrict search to
            use_keyword_extraction: Whether to use extracted keywords for FTS search

        Returns:
            List of lists of chunk ids. Each sub-list corresponds to one query unit and is 
            deduplicated by RRF, but the overall list is not deduplicated across queries.
        """
        logger.debug(f"retrieve_chunk_ids_by_hybrid_search_multi_query_async: "
                     f"queries: {query_units}, document_ids: {document_ids}, subset_ids: {subset_ids}")
        
        # If document_ids filter is provided, convert to chunk_ids filter
        if document_ids:
            subset_ids = self.chunk_dao.get_chunk_ids_by_document_ids(document_ids)
            logger.debug(f"Filtered to {len(subset_ids)} chunks from {len(document_ids)} documents")
        
        # Run all searches concurrently using asyncio.gather
        import asyncio
        return await asyncio.gather(*[
            self.retrieve_chunk_ids_by_hybrid_search_async(query_unit, subset_ids, use_keyword_extraction=use_keyword_extraction) 
            for query_unit in query_units
        ])

    def retrieve_chunk_ids_by_hybrid_search(self, query: RetrievalQueryUnit, subset_ids: List[int] = None, use_keyword_extraction: bool = False) -> List[int]:
        """Retrieve chunks using hybrid search (FTS + semantic search combined with RRF).
        
        Args:
            query: The query unit containing the query string and optional keywords
            subset_ids: Optional list of chunk IDs to restrict search to
            use_keyword_extraction: If True and query has keywords, search each keyword separately
            
        Returns:
            List of chunk IDs ranked by RRF combination of FTS and semantic search
        """
        logger.debug(f"retrieve_chunk_ids_by_hybrid_search: query: {query}, subset_ids: {subset_ids}")
        
        # Perform FTS (Full-Text Search)
        if use_keyword_extraction and query.keywords:
            # Search for each keyword separately and consolidate with RRF
            # Currently treats each keyword equally; weights could be added in the future
            fts_results_list = [self.retrieve_chunk_ids_by_keyword_search(keyword, subset_ids) 
                               for keyword in query.keywords]
            fts_results, _ = self._rrf(fts_results_list)
        else:
            fts_results = self.retrieve_chunk_ids_by_keyword_search(query.query, subset_ids)
            
        # Limit FTS results (keyword extraction may return more than the configured limit)
        fts_results = fts_results[:self.retriever_config.fts_retrieval_limit]

        # Perform semantic (vector) search
        logger.debug(f"fts_results: {fts_results}")
        vec_results = self.retrieve_chunk_ids_by_semantic_search(query.query, subset_ids)
        logger.debug(f"vec_results: {vec_results}")
        
        # Combine FTS and vector results using Reciprocal Rank Fusion
        rrf_results, _ = self._rrf_fts_and_vec(fts_results, vec_results)

        logger.info(f"retrieve_chunk_ids_by_hybrid_search retrieved {len(rrf_results)} ids")
        return rrf_results
    
    async def retrieve_chunk_ids_by_hybrid_search_async(self, query: RetrievalQueryUnit, subset_ids: List[int] = None, use_keyword_extraction: bool = False) -> List[int]:
        """Asynchronously retrieve chunks using hybrid search (FTS + semantic search combined with RRF).
        
        Args:
            query: The query unit containing the query string and optional keywords
            subset_ids: Optional list of chunk IDs to restrict search to
            use_keyword_extraction: If True and query has keywords, search each keyword separately
            
        Returns:
            List of chunk IDs ranked by RRF combination of FTS and semantic search
        """
        logger.debug(f"retrieve_chunk_ids_by_hybrid_search_async: query: {query}, subset_ids: {subset_ids}")
        
        # Perform FTS (Full-Text Search)
        if use_keyword_extraction and query.keywords:
            # Search for each keyword separately and consolidate with RRF
            # Currently treats each keyword equally; weights could be added in the future
            fts_results_list = [self.retrieve_chunk_ids_by_keyword_search(keyword, subset_ids) 
                               for keyword in query.keywords]
            fts_results, _ = self._rrf(fts_results_list)
        else:
            fts_results = self.retrieve_chunk_ids_by_keyword_search(query.query, subset_ids)
            
        # Limit FTS results (keyword extraction may return more than the configured limit)
        fts_results = fts_results[:self.retriever_config.fts_retrieval_limit]

        # Perform semantic (vector) search asynchronously
        logger.debug(f"fts_results: {fts_results}")
        vec_results = await self.retrieve_chunk_ids_by_semantic_search_async(query.query, subset_ids)
        logger.debug(f"vec_results: {vec_results}")
        
        # Combine FTS and vector results using Reciprocal Rank Fusion
        rrf_results, _ = self._rrf_fts_and_vec(fts_results, vec_results)

        logger.info(f"retrieve_chunk_ids_by_hybrid_search_async retrieved {len(rrf_results)} ids")
        return rrf_results

    def consolidate_results(self, results: List[List[any]]) -> List[any]:
        """Consolidate results from multiple query units using Reciprocal Rank Fusion.
        
        Args:
            results: List of result lists, one per query unit
            
        Returns:
            Single consolidated list ranked by RRF scores
        """
        consolidated_results, _ = self._rrf(results)
        return consolidated_results
    
    def _get_chat_context_units_list_from_chunk_ids_list_preserve_order(self, chunk_ids_list: List[List[int]]) -> List[List[ChatContextUnit]]:
        """Convert lists of chunk IDs to lists of chat context units with metadata.
        
        Args:
            chunk_ids_list: List of chunk ID lists (one per query)
            
        Returns:
            List of chat context unit lists with metadata from database
        """
        chat_context_units_list = []
        
        for chunk_ids in chunk_ids_list:
            # Get valid chunk infos (filters out chunks that don't exist)
            chunk_infos = [info for info in (self._get_chunk_info_from_chunk_id(chunk_id) for chunk_id in chunk_ids) if info]
            # Create chat context units with metadata
            chat_context_units = [self._get_chat_context_unit_from_chunk_info(info) for info in chunk_infos]
            chat_context_units_list.append(chat_context_units)
        
        return chat_context_units_list
    
    def _get_chunk_info_from_chunk_id(self, chunk_id: int) -> Optional[ChunkInfo]:
        """Get chunk info from database for a given chunk ID.
        
        Args:
            chunk_id: The ID of the chunk to retrieve
            
        Returns:
            ChunkInfo object if found, None otherwise
        """
        chunk = self.chunk_dao.get_by_id(chunk_id)
        if not chunk:
            logger.warning(f"chunk_id {chunk_id} not found in DB")
            return None
        return ChunkInfo(chunk_id=chunk_id, document_id=chunk.document_id, chunk_content=chunk.chunk_text)
    
    def _get_chat_context_unit_from_chunk_info(self, chunk_info: ChunkInfo) -> ChatContextUnit:
        """Convert a chunk info to a chat context unit with full document metadata.
        
        Args:
            chunk_info: Basic chunk information
            
        Returns:
            ChatContextUnit with full metadata from the database
        """
        document = self.document_dao.get_by_id(chunk_info.document_id)
            
        return ChatContextUnit(
            chunk_id=chunk_info.chunk_id,
            document_id=chunk_info.document_id,
            chunk_source=document.file_name,
            chunk_content=chunk_info.chunk_content,
            document=document,
            current_time=datetime.datetime.now(),
            rel_pos=None  # Relative position not yet implemented
        )

    def _rrf_fts_and_vec(self, fts_results: List[int], vec_results: List[int]) -> Tuple[List[int], List[float]]:
        """Perform Reciprocal Rank Fusion on FTS and vector search results.
        
        Args:
            fts_results: Sorted chunk IDs from FTS search (highest match first)
            vec_results: Sorted chunk IDs from vector search (highest match first)
            
        Returns:
            Tuple of (sorted_ids, rrf_scores) sorted by RRF score descending
        """
        logger.debug(f"_rrf_fts_and_vec: fts_results: {fts_results}, vec_results: {vec_results}")
        return self._rrf([fts_results, vec_results])

    def _rrf(self, results: List[List[any]]) -> Tuple[List[any], List[float]]:
        """Apply Reciprocal Rank Fusion to combine multiple ranked result lists.
        
        Uses the RRF utility function from utils.rrf.
        RRF formula: score(id) = sum(1 / (k + rank)) for each list the id appears in
        
        Args:
            results: List of ranked result lists to combine
            
        Returns:
            Tuple of (sorted_ids, rrf_scores) sorted by RRF score descending
        """
        k = self.retriever_config.reciprocal_rank_fusion_k
        sorted_results_with_scores = rrf_with_scores(results, k=k)
        
        # Separate IDs and scores
        sorted_ids = [item_id for item_id, _ in sorted_results_with_scores]
        scores = [score for _, score in sorted_results_with_scores]
        
        return sorted_ids, scores

    def _get_readable_chunk_results(self, chunk_id: int):
        """Get a readable representation of a chunk result for debugging/logging."""
        chunk = self.chunk_dao.get_by_id(chunk_id)
        document_id = chunk.document_id
        document_name = self.document_dao.get_by_id(document_id).file_name
        return f"[Chunk-ID-{chunk_id}][Doc-ID-{document_id}][Document: {document_name}]: {repr(chunk.chunk_text[:200])}..."

    def _sort_by_scores(self, scores: list, *item_lists, reverse: bool = True) -> tuple:
        """
        Sort multiple lists based on corresponding scores.
        
        Args:
            scores: List of scores used for sorting
            *item_lists: Arbitrary number of lists to be sorted based on scores
            reverse: Whether to sort in descending order (default: True)
            
        Returns:
            tuple: (sorted_scores, sorted_list1, sorted_list2, ...)
        """
        # Validate all lists have the same length as scores
        for i, item_list in enumerate(item_lists):
            if len(item_list) != len(scores):
                raise ValueError(f"List at position {i} has length {len(item_list)}, but scores has length {len(scores)}")
            
        # Create list of indices
        indices = list(range(len(scores)))
        
        # Sort indices based on scores
        sorted_indices = sorted(indices, key=lambda i: scores[i], reverse=reverse)
        
        # Sort all lists using the sorted indices
        sorted_scores = [scores[i] for i in sorted_indices]
        sorted_lists = [[item_list[i] for i in sorted_indices] for item_list in item_lists]
        
        # Return the sorted scores and all sorted lists
        return (sorted_scores, *sorted_lists)
        
    def _rerank(
        self,
        query_units: List[RetrievalQueryUnit],
        chat_context_units_list: List[List[ChatContextUnit]],
    ) -> List[List[ChatContextUnit]]:
        """
        Rerank chat context units using the reranker service.

        Args:
            query_units: List of retrieval query units
            chat_context_units_list: List of lists of chat context units to rerank

        Returns:
            List[List[ChatContextUnit]]: Reranked lists of chat context units
        """
        reranked_chat_context_units_list = []
        logger.debug(
            f"queries before reranking: {[[chat_context_unit.chunk_id for chat_context_unit in chat_context_units] for chat_context_units in chat_context_units_list]}"
        )

        # Calculate items to rerank per list
        if self.config.max_item_to_rerank is not None:
            items_to_rerank_per_list = self.config.max_item_to_rerank // len(
                query_units
            )
        else:
            items_to_rerank_per_list = None

        for query_unit, chat_context_units in zip(
            query_units, chat_context_units_list
        ):
            context_with_metadata = list(map(str, chat_context_units))
            if (
                items_to_rerank_per_list is not None
                and items_to_rerank_per_list >= 0
            ):
                if items_to_rerank_per_list < len(context_with_metadata):
                    logger.info(
                        f"Truncating {len(context_with_metadata)} to {items_to_rerank_per_list} for reranking"
                    )

                context_with_metadata = context_with_metadata[
                    :items_to_rerank_per_list
                ]
                chat_context_units = chat_context_units[
                    :items_to_rerank_per_list
                ]

            # TODO: tune this batch size
            scores = self.reranker.rerank(
                query_unit.query, context_with_metadata, batch_size=1
            )

            for score, chat_context_unit in zip(scores, chat_context_units):
                chat_context_unit.rerank_score = score

            scores, chat_context_units = self._sort_by_scores(
                scores, chat_context_units
            )

            # Initialize cutoff_index to keep all items by default
            cutoff_index = len(scores)
            logger.debug(
                f"before rerank filtering: {scores}, {[chat_context_unit.chunk_id for chat_context_unit in chat_context_units]}"
            )

            # Cutoff criteria 1: Cumulative probability threshold
            cumulative_cutoff_index = len(scores)
            cumulative_prob = 1.0
            for i, score in enumerate(scores):
                cumulative_prob -= score

                # small buffer to avoid floating point precision issues
                if cumulative_prob < -1e-6 or cumulative_prob > 1.0 + 1e-6:
                    raise ValueError(
                        f"Cumulative probability is out of range: {cumulative_prob}, there must be a bug"
                    )

                if cumulative_prob < self.config.rerank_keep_threshold:
                    cumulative_cutoff_index = (
                        i + 1
                    )  # Keep this one, this also handles the case
                    break

            # Cutoff criteria 2: Score cliff (large drop from first score)
            cliff_cutoff_index = len(scores)
            if len(scores) > 0:
                first_score = scores[0]
                for i, score in enumerate(scores):
                    if (
                        first_score - score
                        > self.config.rerank_cliff_cutoff_score_difference
                    ):
                        cliff_cutoff_index = (
                            i  # Do not keep this one off the plateau
                        )
                        break

            # Priority: if cliff is detected, use it; otherwise use cumulative
            if cliff_cutoff_index < len(scores):
                # Cliff detected - use cliff cutoff
                cutoff_index = cliff_cutoff_index
                cutoff_reason = "cliff"
                logger.info(
                    f"Cliff detected: keeping {cutoff_index} items before score drop"
                )
            else:
                # No cliff - use cumulative probability cutoff
                cutoff_index = cumulative_cutoff_index
                if cutoff_index == len(scores):
                    logger.info(
                        f"All items meet both rerank thresholds, did we fetch enough items for reranking?"
                    )
                    cutoff_reason = "none"
                else:
                    logger.info(
                        f"No cliff detected, using cumulative threshold: keeping {cutoff_index} items"
                    )
                    cutoff_reason = "cumulative"

            chat_context_units = chat_context_units[:cutoff_index]
            scores = scores[:cutoff_index]
            logger.debug(
                f"after rerank filtering: {scores}, {[chat_context_unit.chunk_id for chat_context_unit in chat_context_units]}"
            )

            # make a copy to avoid losing reference
            reranked_chat_context_units_list.append(chat_context_units[:])

        logger.debug(
            f"queries after reranking: {[[chat_context_unit.chunk_id for chat_context_unit in chat_context_units] for chat_context_units in reranked_chat_context_units_list]}"
        )
        return reranked_chat_context_units_list
    
    async def _rerank_async(
        self,
        query_units: List[RetrievalQueryUnit],
        chat_context_units_list: List[List[ChatContextUnit]],
    ) -> List[List[ChatContextUnit]]:
        """
        Asynchronously rerank chat context units using the reranker service.

        Args:
            query_units: List of retrieval query units
            chat_context_units_list: List of lists of chat context units to rerank

        Returns:
            List[List[ChatContextUnit]]: Reranked lists of chat context units
        """
        reranked_chat_context_units_list = []
        logger.debug(
            f"queries before reranking (async): {[[chat_context_unit.chunk_id for chat_context_unit in chat_context_units] for chat_context_units in chat_context_units_list]}"
        )

        # Calculate items to rerank per list
        if self.config.max_item_to_rerank is not None:
            items_to_rerank_per_list = self.config.max_item_to_rerank // len(
                query_units
            )
        else:
            items_to_rerank_per_list = None

        for query_unit, chat_context_units in zip(
            query_units, chat_context_units_list
        ):
            context_with_metadata = list(map(str, chat_context_units))
            if (
                items_to_rerank_per_list is not None
                and items_to_rerank_per_list >= 0
            ):
                if items_to_rerank_per_list < len(context_with_metadata):
                    logger.info(
                        f"Truncating {len(context_with_metadata)} to {items_to_rerank_per_list} for reranking (async)"
                    )

                context_with_metadata = context_with_metadata[
                    :items_to_rerank_per_list
                ]
                chat_context_units = chat_context_units[
                    :items_to_rerank_per_list
                ]

            # TODO: tune this batch size
            scores = await self.reranker.rerank_async(
                query_unit.query, context_with_metadata, batch_size=1
            )

            for score, chat_context_unit in zip(scores, chat_context_units):
                chat_context_unit.rerank_score = score

            scores, chat_context_units = self._sort_by_scores(
                scores, chat_context_units
            )

            # Initialize cutoff_index to keep all items by default
            cutoff_index = len(scores)
            logger.debug(
                f"before rerank filtering (async): {scores}, {[chat_context_unit.chunk_id for chat_context_unit in chat_context_units]}"
            )

            # Cutoff criteria 1: Cumulative probability threshold
            cumulative_cutoff_index = len(scores)
            cumulative_prob = 1.0
            for i, score in enumerate(scores):
                cumulative_prob -= score

                # small buffer to avoid floating point precision issues
                if cumulative_prob < -1e-6 or cumulative_prob > 1.0 + 1e-6:
                    raise ValueError(
                        f"Cumulative probability is out of range: {cumulative_prob}, there must be a bug"
                    )

                if cumulative_prob < self.config.rerank_keep_threshold:
                    cumulative_cutoff_index = (
                        i + 1
                    )  # Keep this one, this also handles the case
                    break

            # Cutoff criteria 2: Score cliff (large drop from first score)
            cliff_cutoff_index = len(scores)
            if len(scores) > 0:
                first_score = scores[0]
                for i, score in enumerate(scores):
                    if (
                        first_score - score
                        > self.config.rerank_cliff_cutoff_score_difference
                    ):
                        cliff_cutoff_index = (
                            i  # Do not keep this one off the plateau
                        )
                        break

            # Priority: if cliff is detected, use it; otherwise use cumulative
            if cliff_cutoff_index < len(scores):
                # Cliff detected - use cliff cutoff
                cutoff_index = cliff_cutoff_index
                cutoff_reason = "cliff"
                logger.info(
                    f"Cliff detected (async): keeping {cutoff_index} items before score drop"
                )
            else:
                # No cliff - use cumulative probability cutoff
                cutoff_index = cumulative_cutoff_index
                if cutoff_index == len(scores):
                    logger.info(
                        f"All items meet both rerank thresholds (async), did we fetch enough items for reranking?"
                    )
                    cutoff_reason = "none"
                else:
                    logger.info(
                        f"No cliff detected (async), using cumulative threshold: keeping {cutoff_index} items"
                    )
                    cutoff_reason = "cumulative"

            chat_context_units = chat_context_units[:cutoff_index]
            scores = scores[:cutoff_index]
            logger.debug(
                f"after rerank filtering (async): {scores}, {[chat_context_unit.chunk_id for chat_context_unit in chat_context_units]}"
            )

            # make a copy to avoid losing reference
            reranked_chat_context_units_list.append(chat_context_units[:])

        logger.debug(
            f"queries after reranking (async): {[[chat_context_unit.chunk_id for chat_context_unit in chat_context_units] for chat_context_units in reranked_chat_context_units_list]}"
        )
        return reranked_chat_context_units_list