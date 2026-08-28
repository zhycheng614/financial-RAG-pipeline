
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
from typing import List, Tuple
from models.base import Base
from dao.base_dao import BaseDao
from models.chunk import Chunk
from models.document import Document
from data_classes.bm25_query import BM25Query
import sqlite3
import logging

# English stopwords for query optimization
ENGLISH_STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
    'to', 'was', 'will', 'with'
}

logger = logging.getLogger(__name__)

class ChunkDao(BaseDao[Chunk]):
    # higher is more important
    FILE_NAME_WEIGHT_IN_SEARCH = 1.5
    
    def __init__(self, session_factory: sessionmaker):
        super().__init__(session_factory, Chunk)
    
    def get_chunks_by_document_id(self, document_id: int) -> List[Chunk]:
        """Get all chunks for a specific document."""
        with self.session_scope() as session:
            return session.query(Chunk).filter(Chunk.document_id == document_id).all()
    
    def delete_chunks_by_document_id(self, document_id: int) -> int:
        """Delete all chunks for a specific document and return the count of deleted chunks."""
        with self.session_scope() as session:
            count = session.query(Chunk).filter(Chunk.document_id == document_id).delete()
            return count
    
    def delete_chunks_by_document_ids(self, document_ids: List[int]) -> int:
        """Bulk delete all chunks for multiple documents and return the count of deleted chunks.
        
        Args:
            document_ids: List of document IDs whose chunks should be deleted
            
        Returns:
            Count of deleted chunks
        """
        if not document_ids:
            return 0
            
        with self.session_scope() as session:
            count = session.query(Chunk).filter(Chunk.document_id.in_(document_ids)).delete(synchronize_session=False)
            return count
    
    def get_chunk_ids_by_document_ids(self, document_ids: List[int] = None) -> List[int]:
        """Get only the ids of chunks matching the given document IDs.
        
        Args:
            document_ids: A list of document IDs to filter by. If None, returns all chunk IDs.
                         If empty list, returns empty list.
            
        Returns:
            List of chunk IDs as integers.
        """
        if document_ids == []:
            return []
        
        with self.session_scope() as session:
            query = session.query(Chunk.id)
            
            if document_ids is not None:
                query = query.filter(Chunk.document_id.in_(document_ids))
            
            result = query.all()
            return [row[0] for row in result]
    
    def get_chunks_by_document_ids(self, document_ids: List[int] = None) -> List[Chunk]:
        """Get all chunks for a list of document IDs.
        
        Args:
            document_ids: A list of document IDs to filter by. If None, returns an empty list.
            
        Returns:
            List of Chunk objects.
        """
        if not document_ids:
            return []
        
        with self.session_scope() as session:
            return session.query(Chunk).filter(Chunk.document_id.in_(document_ids)).all()
        
    def get_chunk_ids_by_file_path(self, file_path: str) -> List[int]:
        """Get all chunk IDs for a specific file path."""
        with self.session_scope() as session:
            result = session.query(Chunk.id)\
                .join(Document, Chunk.document_id == Document.id)\
                .filter(Document.document_path == file_path)\
                .all()
            
            return [row[0] for row in result]

    def _generate_fts5_queries(self, input_query: str) -> List[BM25Query]:
        """
        Generate different variations of the query for FTS5.
        Handles both English and Chinese queries without external tokenizer dependencies.
        """
        
        # Escape double quotes by doubling them
        escaped_input = input_query.replace('"', '""')
        
        queries = []
                
        # English query processing
        # Split into words
        words = [w for w in escaped_input.split() if w]
        
        if not words:
            # Return empty query if no words found
            return []

        # Exact match query for the entire phrase
        exact_query = BM25Query(
            input=" ".join(words),
            query=f'"{" ".join(words)}"',
            is_exact=True,
            q_length=len(words),
            i_length=len(words),
            r_length=len(words)
        )
        queries.append(exact_query)

        # Create quoted terms for non-stopwords
        quoted_terms = [f'"{word}"' for word in words if word.lower() not in ENGLISH_STOP_WORDS]

        if quoted_terms:
            # OR query with all non-stopword terms
            or_query = BM25Query(
                input=" ".join(words),
                query=f'({" OR ".join(quoted_terms)})',
                is_exact=False,
                q_length=1,  # length of phrase
                i_length=len(words),
                r_length=len(words) - len(quoted_terms)
            )
            queries.append(or_query)
        else:
            # Fallback if all words are stopwords
            regular_query = BM25Query(
                input=" ".join(words),
                query=" ".join(words),
                is_exact=False,
                q_length=len(words),
                i_length=len(words),
                r_length=len(words)
            )
            queries.append(regular_query)

        return queries

    def search_with_scores(self, query: str, subset_ids: List[int] = None, limit: int = 10) -> Tuple[List[int], List[float], str]:
        """
        Search for chunks matching the given query using BM25 ranking with SQLAlchemy.
        
        Args:
            query: The search query
            subset_ids: List of chunk IDs to constrain search to
            limit: Maximum number of results to return
            
        Returns:
            Tuple containing:
            - List of chunk IDs sorted by relevance
            - List of BM25 scores
            - The query string that was used to produce results
        """
        # If subset_ids is an empty list, return empty results immediately
        if subset_ids is not None and len(subset_ids) == 0:
            return [], [], None
            
        # Generate BM25 queries
        with self.session_scope() as session:
            bm25_queries = self._generate_fts5_queries(query)
            
            chunk_ids = []
            scores = []
            matched_query = None
            
            for bm25_query in bm25_queries:
                # Build the SQL query based on subset_ids
                if subset_ids is not None and len(subset_ids) > 0:
                    chunks_str = ", ".join(str(id) for id in subset_ids)
                    sql = text(f"""
                    SELECT fts.id, bm25(chunks_fts) as score
                    FROM chunks_fts fts
                    WHERE chunks_fts MATCH :query
                    AND fts.id IN ({chunks_str})
                    ORDER BY score ASC
                    LIMIT :limit
                    """)
                else:
                    sql = text(f"""
                    SELECT fts.id, bm25(chunks_fts) as score
                    FROM chunks_fts fts
                    WHERE chunks_fts MATCH :query
                    ORDER BY score ASC
                    LIMIT :limit
                    """)
                
                # Execute the query with parameters
                results = session.execute(sql, {'query': bm25_query.query, 'limit': limit}).fetchall()
                
                if results:
                    # Save the query that was used to produce results
                    matched_query = bm25_query.query
                    # Results are already sorted by SQL
                    chunk_ids = [row[0] for row in results]
                    scores = [row[1] for row in results]
                    break
                
            return chunk_ids, scores, matched_query
        

    def search(self, query: str, subset_ids: List[int] = None, limit: int = 10) -> Tuple[List[int], str]:
        """
        Search for chunks matching the given query using BM25 ranking with SQLAlchemy.
        
        Args:
            query: The search query
            subset_ids: List of chunk IDs to constrain search to
            limit: Maximum number of results to return
            
        Returns:
            Tuple containing:
            - List of chunk IDs sorted by relevance (best matches first)
            - The query string that was used to produce results
        """
        chunk_ids, _, matched_query = self.search_with_scores(
            query, 
            subset_ids=subset_ids,
            limit=limit
        )
        return chunk_ids, matched_query