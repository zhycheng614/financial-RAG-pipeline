#!/usr/bin/env python
"""
Concurrent query processing pipeline with RAG answer generation.

This script:
1. Takes queries from multiple sources (CLI, CSV, or HuggingFace dataset)
2. Uses async logic to process queries and retrieve relevant chunks
3. Generates final answers using LLM with retrieved context
4. Outputs results to terminal or CSV file
"""

import argparse
import asyncio
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, AsyncGenerator
from dataclasses import dataclass, field

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from beans.embedder import Embedder
from beans.faiss_manager import FaissIndex
from beans.sqlite3_db_manager import Sqlite3DbManager
from beans.reranker import JinaRerankService
from beans.local_reranker import LocalRerankService
from beans.inference_facade import InferenceFacade
from dao.document_dao import DocumentDao
from dao.chunk_dao import ChunkDao
from service.query_processor import QueryProcessor
from service.rewriting_service import RewritingService
from service.vector_search_service import VectorSearchService
from data_classes.embedding_generator_config import EmbedderConfig
from data_classes.query_config import QueryConfig
from data_classes.retriever_config import RetrieverConfig
from data_classes.chat_context_unit import ChatContextUnit
from data_classes.inference_parameters import InferenceParameters
from prompts import RAG_ANSWER_GENERATION_SYSTEM_PROMPT, RAG_ANSWER_GENERATION_USER_PROMPT
from constants import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_SEMANTIC_FILTERING_THRESHOLD,
    DEFAULT_FINAL_CHUNKS_TO_ASSEMBLE_LIMIT,
    DEFAULT_FTS_RETRIEVAL_LIMIT,
    DEFAULT_SEMANTIC_RETRIEVAL_LIMIT,
    DEFAULT_RRF_K,
    DEFAULT_RERANK_KEEP_THRESHOLD,
    DEFAULT_RERANK_CLIFF_CUTOFF_SCORE_DIFFERENCE,
    DEFAULT_MAX_ITEM_TO_RERANK,
    DEFAULT_LOCAL_RERANKER_MODEL_PATH,
    DEFAULT_LOCAL_RERANKER_TOKENIZER_PATH,
    DEFAULT_LOCAL_RERANKER_NUM_WORKERS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default database and index file names (in project root)
DEFAULT_DB_NAME = "rag_pipeline.db"
DEFAULT_FAISS_INDEX_NAME = "faiss_index.index"

# Default inference settings
DEFAULT_MODEL = "gpt-4.1"
DEFAULT_BASE_URL = "https://api.openai.com/v1"


def get_project_paths(project_name: Optional[str] = None) -> Tuple[str, str]:
    """
    Get database and FAISS index paths based on project name.
    
    Args:
        project_name: Optional project name. If provided, paths will be 
                     {project_name}.rag_pipeline.db and {project_name}.faiss_index.index
                     
    Returns:
        Tuple of (db_path, faiss_index_path)
    """
    if project_name:
        db_path = f"{project_name}.rag_pipeline.db"
        faiss_index_path = f"{project_name}.faiss_index.index"
    else:
        db_path = DEFAULT_DB_NAME
        faiss_index_path = DEFAULT_FAISS_INDEX_NAME
    
    return db_path, faiss_index_path


@dataclass
class QueryItem:
    """Represents a single query with optional ID."""
    query: str
    query_id: Optional[str] = None
    row_index: int = 0


@dataclass
class QueryResult:
    """Represents the result of processing a single query."""
    query_item: QueryItem
    chat_context_units: List[ChatContextUnit] = field(default_factory=list)
    final_answer: str = ""
    error: Optional[str] = None
    
    def get_chunk_ids(self) -> List[int]:
        """Get list of chunk IDs from chat context units."""
        return [unit.chunk_id for unit in self.chat_context_units]
    
    def get_document_ids(self) -> List[int]:
        """Get list of unique document IDs from chat context units."""
        return list(set(unit.document_id for unit in self.chat_context_units))


class QueryInputSource:
    """Base class for query input sources."""
    
    async def get_queries(self) -> List[QueryItem]:
        """Get list of query items from the input source."""
        raise NotImplementedError


class CLIInputSource(QueryInputSource):
    """Input source for single query from CLI."""
    
    def __init__(self, query: str):
        self.query = query
    
    async def get_queries(self) -> List[QueryItem]:
        """Get single query from CLI."""
        return [QueryItem(query=self.query, query_id="cli_query", row_index=0)]


class CSVInputSource(QueryInputSource):
    """Input source for queries from CSV file."""
    
    def __init__(self, csv_path: str, column_name: str, id_column: Optional[str] = None):
        self.csv_path = csv_path
        self.column_name = column_name
        self.id_column = id_column
    
    async def get_queries(self) -> List[QueryItem]:
        """Get queries from CSV file."""
        queries = []
        
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                # Validate column exists
                if self.column_name not in reader.fieldnames:
                    raise ValueError(f"Column '{self.column_name}' not found in CSV. Available columns: {reader.fieldnames}")
                
                # Check if ID column exists
                has_id_column = self.id_column and self.id_column in reader.fieldnames
                
                for row_idx, row in enumerate(reader):
                    query_text = row.get(self.column_name, "").strip()
                    if not query_text:
                        logger.warning(f"Empty query at row {row_idx + 1}, skipping")
                        continue
                    
                    # Try to get ID from specified column, fallback to '_id' or 'id', then row index
                    query_id = None
                    if has_id_column:
                        query_id = row.get(self.id_column)
                    elif '_id' in reader.fieldnames:
                        query_id = row.get('_id')
                    elif 'id' in reader.fieldnames:
                        query_id = row.get('id')
                    
                    if query_id is None:
                        query_id = str(row_idx)
                    
                    queries.append(QueryItem(
                        query=query_text,
                        query_id=str(query_id),
                        row_index=row_idx
                    ))
                    
            logger.info(f"Loaded {len(queries)} queries from CSV file: {self.csv_path}")
            
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            raise
        
        return queries


class HuggingFaceInputSource(QueryInputSource):
    """Input source for queries from HuggingFace dataset."""
    
    def __init__(self, dataset_id: str, column_name: str, split: str = "train", id_column: Optional[str] = None):
        self.dataset_id = dataset_id
        self.column_name = column_name
        self.split = split
        self.id_column = id_column
    
    async def get_queries(self) -> List[QueryItem]:
        """Get queries from HuggingFace dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install the 'datasets' package: pip install datasets")
        
        queries = []
        
        try:
            logger.info(f"Loading HuggingFace dataset: {self.dataset_id}, split: {self.split}")
            dataset = load_dataset(self.dataset_id, split=self.split)
            
            # Validate column exists
            if self.column_name not in dataset.column_names:
                raise ValueError(f"Column '{self.column_name}' not found in dataset. Available columns: {dataset.column_names}")
            
            # Check if ID column exists
            has_id_column = self.id_column and self.id_column in dataset.column_names
            
            for row_idx, row in enumerate(dataset):
                query_text = row.get(self.column_name, "")
                if isinstance(query_text, str):
                    query_text = query_text.strip()
                else:
                    query_text = str(query_text).strip()
                
                if not query_text:
                    logger.warning(f"Empty query at row {row_idx}, skipping")
                    continue
                
                # Try to get ID from specified column, fallback to '_id' or 'id', then row index
                query_id = None
                if has_id_column:
                    query_id = row.get(self.id_column)
                elif '_id' in dataset.column_names:
                    query_id = row.get('_id')
                elif 'id' in dataset.column_names:
                    query_id = row.get('id')
                
                if query_id is None:
                    query_id = str(row_idx)
                
                queries.append(QueryItem(
                    query=query_text,
                    query_id=str(query_id),
                    row_index=row_idx
                ))
            
            logger.info(f"Loaded {len(queries)} queries from HuggingFace dataset: {self.dataset_id}")
            
        except Exception as e:
            logger.error(f"Error loading HuggingFace dataset: {e}")
            raise
        
        return queries


async def process_single_query(
    query_item: QueryItem,
    query_processor: QueryProcessor,
    inference_facade: InferenceFacade
) -> QueryResult:
    """
    Process a single query: retrieve chunks and generate final answer.
    
    Args:
        query_item: Query to process
        query_processor: QueryProcessor instance
        inference_facade: InferenceFacade instance for answer generation
        
    Returns:
        QueryResult with retrieved chunks and final answer
    """
    result = QueryResult(query_item=query_item)
    
    try:
        logger.info(f"Processing query [{query_item.query_id}]: {query_item.query[:100]}...")
        
        # Step 1: Process query to retrieve relevant chunks (now fully async)
        chat_context_units = await query_processor.process_query_async(query_item.query)
        result.chat_context_units = chat_context_units
        
        logger.info(f"Retrieved {len(chat_context_units)} chunks for query [{query_item.query_id}]")
        
        # Step 2: Generate final answer using LLM with retrieved context
        if chat_context_units:
            # Format context from retrieved chunks
            context_str = "\n\n---\n\n".join([
                f"[Chunk {i+1}] {unit.chunk_source}:\n{unit.chunk_content}"
                for i, unit in enumerate(chat_context_units)
            ])
            
            # Prepare messages for LLM
            user_prompt = RAG_ANSWER_GENERATION_USER_PROMPT.format(
                context=context_str,
                query=query_item.query
            )
            
            messages = [
                {"role": "system", "content": RAG_ANSWER_GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
            
            # Generate answer (now fully async)
            logger.info(f"Generating answer for query [{query_item.query_id}]...")
            final_answer = await inference_facade.create_chat_completion_async(messages)
            result.final_answer = final_answer.strip()
            
            logger.info(f"✓ Completed query [{query_item.query_id}]")
        else:
            result.final_answer = "No relevant documents found to answer this question."
            logger.warning(f"No chunks retrieved for query [{query_item.query_id}]")
            
    except Exception as e:
        logger.error(f"✗ Error processing query [{query_item.query_id}]: {str(e)}", exc_info=True)
        result.error = str(e)
        result.final_answer = f"Error: {str(e)}"
    
    return result


async def process_queries_concurrently(
    query_items: List[QueryItem],
    query_processor: QueryProcessor,
    inference_facade: InferenceFacade,
    max_concurrent_tasks: int = 3
) -> List[QueryResult]:
    """
    Process multiple queries concurrently.
    
    Args:
        query_items: List of queries to process
        query_processor: QueryProcessor instance
        inference_facade: InferenceFacade instance
        max_concurrent_tasks: Maximum number of concurrent tasks
        
    Returns:
        List of query results in the same order as input
    """
    # Create semaphore to limit concurrent tasks
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    
    async def process_with_semaphore(query_item: QueryItem) -> QueryResult:
        async with semaphore:
            return await process_single_query(query_item, query_processor, inference_facade)
    
    # Create tasks for all queries
    tasks = [process_with_semaphore(query_item) for query_item in query_items]
    
    # Run all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle exceptions and convert to QueryResult
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Exception in query processing: {result}")
            final_results.append(QueryResult(
                query_item=query_items[i],
                error=str(result),
                final_answer=f"Error: {str(result)}"
            ))
        else:
            final_results.append(result)
    
    return final_results


def output_to_terminal(results: List[QueryResult]):
    """Output results to terminal."""
    print("\n" + "=" * 80)
    print("QUERY RESULTS")
    print("=" * 80)
    
    for i, result in enumerate(results, 1):
        print(f"\n[Query {i}] ID: {result.query_item.query_id}")
        print(f"Question: {result.query_item.query}")
        print(f"Retrieved Chunks: {len(result.chat_context_units)}")
        if result.chat_context_units:
            print(f"Chunk IDs: {result.get_chunk_ids()}")
            print(f"Document IDs: {result.get_document_ids()}")
        print(f"\nAnswer:\n{result.final_answer}")
        print("-" * 80)


def output_to_csv(results: List[QueryResult], output_path: str):
    """Output results to CSV file."""
    try:
        # Create parent directories if they don't exist
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file already exists and prompt user
        if output_file.exists():
            print(f"\n⚠ File already exists: {output_path}")
            override = input("Override existing file? (y/n, default: n): ").strip().lower()
            
            if override != 'y':
                # Add timestamp to filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Split filename and extension
                stem = output_file.stem
                suffix = output_file.suffix
                # Create new filename with timestamp
                new_filename = f"{stem}_{timestamp}{suffix}"
                output_path = str(output_file.parent / new_filename)
                output_file = Path(output_path)
                print(f"✓ Saving to new file: {output_path}")
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['query_id', 'query', 'chunk_ids', 'document_ids', 'final_answer']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                # Format query and answer as single-line strings by replacing newlines
                formatted_query = result.query_item.query.replace('\n', ' ').replace('\r', ' ')
                formatted_query = ' '.join(formatted_query.split())
                
                formatted_answer = result.final_answer.replace('\n', ' ').replace('\r', ' ')
                formatted_answer = ' '.join(formatted_answer.split())
                
                writer.writerow({
                    'query_id': result.query_item.query_id,
                    'query': formatted_query,
                    'chunk_ids': ','.join(map(str, result.get_chunk_ids())),
                    'document_ids': ','.join(map(str, result.get_document_ids())),
                    'final_answer': formatted_answer
                })
        
        logger.info(f"Results written to CSV file: {output_path}")
        print(f"\n✓ Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error writing to CSV file: {e}")
        raise


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='RAG Pipeline - Concurrent Query Processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Single query from CLI
  python main/main_query.py --query "What is machine learning?"
  
  # Query with custom project name
  python main/main_query.py --query "What is machine learning?" --project-name myproject
  
  # Queries from CSV file
  python main/main_query.py --csv queries.csv --column question
  
  # Queries from CSV with ID column
  python main/main_query.py --csv queries.csv --column question --id-column query_id
  
  # Queries from HuggingFace dataset
  python main/main_query.py --hf-dataset squad --column question --split validation
  
  # Output to CSV
  python main/main_query.py --csv queries.csv --column question --output-csv results.csv
  
  # Adjust concurrency
  python main/main_query.py --csv queries.csv --column question --concurrent 5

Input Options:
  • CLI: Single query provided via --query argument
  • CSV: Multiple queries from a CSV file column
  • HuggingFace: Multiple queries from a HuggingFace dataset column

Output Options:
  • Terminal: Default, prints results to console
  • CSV: Save results to CSV file with --output-csv

Configuration:
  • Use --project-name to query against specific project database/index
  • Use --no-rewrite to disable query rewriting
  • Use --no-rerank to disable reranking
  • Use --rerank-use-local to use local reranking (nexaai) instead of cloud Jina API
  • Use --rerank-local-workers to set number of worker processes for local reranking
  • Use --no-retrieve to skip retrieval (testing only)
        '''
    )
    
    # Input source arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--query',
        type=str,
        help='Single query from command line'
    )
    input_group.add_argument(
        '--csv',
        type=str,
        metavar='PATH',
        help='Path to CSV file containing queries'
    )
    input_group.add_argument(
        '--hf-dataset',
        type=str,
        metavar='DATASET_ID',
        help='HuggingFace dataset ID'
    )
    
    # Project configuration
    parser.add_argument(
        '-p', '--project-name',
        type=str,
        default=None,
        metavar='NAME',
        help='Project name for database and index files (e.g., "myproject" uses myproject.rag_pipeline.db and myproject.faiss_index.index)'
    )
    
    # Column and split arguments
    parser.add_argument(
        '--column',
        type=str,
        metavar='NAME',
        help='Column name containing queries (required for --csv and --hf-dataset)'
    )
    parser.add_argument(
        '--id-column',
        type=str,
        metavar='NAME',
        help='Column name for query IDs (optional, defaults to row index)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        metavar='NAME',
        help='Dataset split for HuggingFace datasets (default: train)'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-csv',
        type=str,
        metavar='PATH',
        help='Output results to CSV file'
    )
    
    # Processing configuration
    parser.add_argument(
        '--concurrent',
        type=int,
        default=3,
        metavar='N',
        help='Number of concurrent query processing tasks (default: 3)'
    )
    
    # Model configuration
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL,
        metavar='NAME',
        help=f'Model name for answer generation (default: {DEFAULT_MODEL})'
    )
    parser.add_argument(
        '--base-url',
        type=str,
        default=DEFAULT_BASE_URL,
        metavar='URL',
        help=f'Base URL for inference service (default: {DEFAULT_BASE_URL})'
    )
    
    # Query processing flags
    parser.add_argument(
        '--no-rewrite',
        action='store_true',
        help='Disable query rewriting'
    )
    parser.add_argument(
        '--no-rerank',
        action='store_true',
        help='Disable reranking'
    )
    parser.add_argument(
        '--rerank-use-local',
        action='store_true',
        help='Use local reranking with nexaai instead of cloud Jina API'
    )
    parser.add_argument(
        '--rerank-local-workers',
        type=int,
        default=DEFAULT_LOCAL_RERANKER_NUM_WORKERS,
        metavar='N',
        help=f'Number of worker processes for local reranking (default: {DEFAULT_LOCAL_RERANKER_NUM_WORKERS})'
    )
    parser.add_argument(
        '--no-retrieve',
        action='store_true',
        help='Disable retrieval (for testing)'
    )
    
    # Logging
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )
    
    return parser.parse_args()


def initialize_services(
    db_path: str, 
    faiss_index_path: str,
    use_local_reranker: bool = False,
    local_reranker_num_workers: int = DEFAULT_LOCAL_RERANKER_NUM_WORKERS
):
    """
    Initialize all required services and return them.
    
    Args:
        db_path: Path to the SQLite database
        faiss_index_path: Path to the FAISS index file
        use_local_reranker: If True, use local reranking with nexaai instead of cloud Jina API
        local_reranker_num_workers: Number of worker processes for local reranking
        
    Returns:
        Dictionary of initialized services
    """
    logger.info("Initializing services...")
    logger.info(f"Database: {db_path}")
    logger.info(f"FAISS Index: {faiss_index_path}")
    
    # Set up database
    schema_path = str(project_root / "schemas" / "fts_schema.sql")
    db_manager = Sqlite3DbManager(
        db_path=db_path,
        schema_file=schema_path
    )
    Session = db_manager.session_factory
    
    # Initialize DAOs
    document_dao = DocumentDao(Session)
    chunk_dao = ChunkDao(Session)
    
    # Initialize embedder
    embedder_config = EmbedderConfig()
    embedder = Embedder(embedder_config)
    
    # Initialize FAISS index
    faiss_index = FaissIndex(
        index_file=faiss_index_path,
        dimension=DEFAULT_EMBEDDING_DIM,
        faiss_lock=None
    )
    faiss_index.load_or_build_from_scratch()
    
    # Initialize vector search service
    vector_search_service = VectorSearchService(
        embedder=embedder,
        faiss_index=faiss_index,
        default_limit=DEFAULT_SEMANTIC_RETRIEVAL_LIMIT
    )
    
    # Initialize reranker (local or cloud)
    if use_local_reranker:
        logger.info("Using LOCAL reranker with nexaai")
        logger.info(f"Model path: {DEFAULT_LOCAL_RERANKER_MODEL_PATH}")
        logger.info(f"Tokenizer path: {DEFAULT_LOCAL_RERANKER_TOKENIZER_PATH}")
        logger.info(f"Number of workers: {local_reranker_num_workers}")
        
        # Get absolute paths for local model files (local_llms/ lives at the
        # actual project root, one level above src/).
        repo_root = project_root.parent
        model_path = str(repo_root / DEFAULT_LOCAL_RERANKER_MODEL_PATH)
        tokenizer_path = str(repo_root / DEFAULT_LOCAL_RERANKER_TOKENIZER_PATH)
        
        reranker = LocalRerankService(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            num_workers=local_reranker_num_workers
        )
    else:
        logger.info("Using CLOUD reranker (Jina API)")
        reranker = JinaRerankService()
    
    logger.info("Services initialized successfully")
    
    return {
        'document_dao': document_dao,
        'chunk_dao': chunk_dao,
        'embedder': embedder,
        'faiss_index': faiss_index,
        'vector_search_service': vector_search_service,
        'reranker': reranker
    }


async def main():
    """Main entry point for the query processing pipeline."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("=" * 80)
    print("RAG PIPELINE - CONCURRENT QUERY PROCESSING")
    print("=" * 80)
    
    # Get database and FAISS index paths based on project name
    db_path, faiss_index_path = get_project_paths(args.project_name)
    
    if args.project_name:
        logger.info(f"Project name: {args.project_name}")
        logger.info(f"Database: {db_path}")
        logger.info(f"FAISS Index: {faiss_index_path}")
    else:
        logger.info(f"Using default paths:")
        logger.info(f"Database: {db_path}")
        logger.info(f"FAISS Index: {faiss_index_path}")
    
    # Validate arguments
    if (args.csv or args.hf_dataset) and not args.column:
        logger.error("--column is required when using --csv or --hf-dataset")
        sys.exit(1)
    
    # Create input source
    if args.query:
        input_source = CLIInputSource(args.query)
        logger.info("Input source: CLI (single query)")
    elif args.csv:
        input_source = CSVInputSource(args.csv, args.column, args.id_column)
        logger.info(f"Input source: CSV file ({args.csv}, column: {args.column})")
    elif args.hf_dataset:
        input_source = HuggingFaceInputSource(
            args.hf_dataset, 
            args.column, 
            args.split,
            args.id_column
        )
        logger.info(f"Input source: HuggingFace dataset ({args.hf_dataset}, column: {args.column}, split: {args.split})")
    else:
        logger.error("No input source specified")
        sys.exit(1)
    
    # Get queries from input source
    try:
        query_items = await input_source.get_queries()
    except Exception as e:
        logger.error(f"Failed to load queries: {e}")
        sys.exit(1)
    
    if not query_items:
        logger.warning("No queries found in input source")
        return
    
    logger.info(f"Loaded {len(query_items)} queries")
    
    # Initialize services (with local or cloud reranker based on CLI flag)
    services = initialize_services(
        db_path, 
        faiss_index_path,
        use_local_reranker=args.rerank_use_local,
        local_reranker_num_workers=args.rerank_local_workers
    )
    
    # Create query configuration
    # When reranking is disabled, always use top 10 chunks
    final_chunks_limit = DEFAULT_FINAL_CHUNKS_TO_ASSEMBLE_LIMIT
    
    retriever_config = RetrieverConfig(
        semantic_filtering_threshold=DEFAULT_SEMANTIC_FILTERING_THRESHOLD,
        final_chunks_to_assemble_limit=final_chunks_limit,
        fts_retrieval_limit=DEFAULT_FTS_RETRIEVAL_LIMIT,
        semantic_retrieval_limit=DEFAULT_SEMANTIC_RETRIEVAL_LIMIT,
        reciprocal_rank_fusion_k=DEFAULT_RRF_K
    )
    
    query_config = QueryConfig(
        retriever_config=retriever_config,
        rerank_keep_threshold=DEFAULT_RERANK_KEEP_THRESHOLD,
        rerank_cliff_cutoff_score_difference=DEFAULT_RERANK_CLIFF_CUTOFF_SCORE_DIFFERENCE,
        run_rewrite=not args.no_rewrite,
        run_retrieve=not args.no_retrieve,
        run_rerank=not args.no_rerank,
        run_with_keyword_extraction=not args.no_rewrite,
        max_item_to_rerank=DEFAULT_MAX_ITEM_TO_RERANK
    )
    
    # Initialize inference facade for answer generation
    inference_facade = InferenceFacade(
        model=args.model,
        base_url=args.base_url
    )
    
    # Initialize rewriting service if needed
    rewriting_service = None
    if query_config.run_rewrite:
        rewriting_service = RewritingService(inference_facade)
    
    # Create query processor
    query_processor = QueryProcessor(
        config=query_config,
        chunk_dao=services['chunk_dao'],
        document_dao=services['document_dao'],
        faiss_index=services['faiss_index'],
        embedder=services['embedder'],
        reranker=services['reranker'] if query_config.run_rerank else None,
        inference_service=inference_facade if query_config.run_rewrite else None,
        rewriting_service=rewriting_service,
        vector_search_service=services['vector_search_service']
    )
    
    # Process queries concurrently
    logger.info(f"\nProcessing {len(query_items)} queries with {args.concurrent} concurrent tasks...")
    print(f"\nProcessing {len(query_items)} queries...")
    
    results = await process_queries_concurrently(
        query_items,
        query_processor,
        inference_facade,
        max_concurrent_tasks=args.concurrent
    )
    
    # Output results
    if args.output_csv:
        output_to_csv(results, args.output_csv)
    else:
        output_to_terminal(results)
    
    # Summary
    successful = sum(1 for r in results if not r.error)
    failed = len(results) - successful
    
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total queries:  {len(results)}")
    print(f"Successful:     {successful}")
    print(f"Failed:         {failed}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

