#!/usr/bin/env python
"""
Hybrid SFR+CBR query processing pipeline.

This pipeline combines Semantic File Routing (SFR) with Chunk-Based Retrieval (CBR)
to achieve both low failure rates and high answer precision:

  Stage 1 (SFR): Extract ticker/year from the query via LLM structured output,
                 then resolve to document IDs in the existing index database.
                 This eliminates cross-document chunk confusion.

  Stage 2 (CBR): Run hybrid search (FTS + semantic + RRF + reranking) scoped
                 to only the chunks belonging to the identified document(s).
                 This provides targeted, precise context for answer generation.

  Stage 3:      Generate the answer from the retrieved chunks using the LLM.

The pipeline reuses the existing indexing artifacts (SQLite DB + FAISS index)
and requires no additional indexing step.
"""

import argparse
import asyncio
import csv
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

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
from main.query_io import (
    QueryItem,
    QueryResult,
    create_input_source,
    output_to_terminal,
)
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
    DEFAULT_LOCAL_RERANKER_NUM_WORKERS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_DB_NAME = "rag_pipeline.db"
DEFAULT_FAISS_INDEX_NAME = "faiss_index.index"
DEFAULT_MODEL = "gpt-4.1"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_YEAR = datetime.now().year

SUPPORTED_EXTENSIONS = [".pdf", ".txt"]


# ---------------------------------------------------------------------------
# Streaming CSV writer (reused from main_filename_based_solution.py)
# ---------------------------------------------------------------------------


class StreamingCSVWriter:
    """Thread-safe CSV writer that writes results incrementally."""

    def __init__(self, filepath: str, fieldnames: List[str]):
        self.filepath = filepath
        self.fieldnames = fieldnames
        self.lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self):
        async with self.lock:
            if not self._initialized:
                Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)
                with open(self.filepath, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=self.fieldnames, quoting=csv.QUOTE_ALL)
                    writer.writeheader()
                self._initialized = True
                logger.info(f"Initialized CSV output file: {self.filepath}")

    @staticmethod
    def _sanitize_text(value) -> str:
        if value is None:
            return ""
        s = str(value)
        s = s.replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n")
        unicode_map = {
            "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
            "\u2013": "-", "\u2014": "--", "\u00d7": "x", "\u00f7": "/",
            "\u2212": "-", "\u00b1": "+/-", "\u2026": "...", "\u2022": "-",
            "\u00b0": " deg", "\u00a0": " ",
        }
        for uc, ac in unicode_map.items():
            s = s.replace(uc, ac)
        return s

    async def write_result(self, result: QueryResult, original_row: int):
        async with self.lock:
            row = {
                "original_row": original_row,
                "query_id": self._sanitize_text(result.query_item.query_id),
                "query": self._sanitize_text(result.query_item.query),
                "tickers": self._sanitize_text(result.metadata.get("tickers", "")),
                "routed_documents": self._sanitize_text(result.metadata.get("routed_documents", "")),
                "chunk_ids": self._sanitize_text(result.metadata.get("chunk_ids", "")),
                "document_ids": self._sanitize_text(result.metadata.get("document_ids", "")),
                "routing_status": self._sanitize_text(result.metadata.get("routing_status", "")),
                "final_answer": self._sanitize_text(result.final_answer),
                "error": self._sanitize_text(result.error),
            }
            row = {k: v for k, v in row.items() if k in self.fieldnames}
            with open(self.filepath, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames, quoting=csv.QUOTE_ALL)
                writer.writerow(row)


# ---------------------------------------------------------------------------
# Document routing helpers
# ---------------------------------------------------------------------------


def find_document_file(base_dir: str, year: int, ticker: str) -> Optional[str]:
    """Find a document file for a given ticker and year, trying supported extensions."""
    for ext in SUPPORTED_EXTENSIONS:
        path = os.path.join(base_dir, str(year), f"{ticker}{ext}")
        if os.path.exists(path) and os.path.isfile(path):
            return path
    return None


def resolve_document_ids(
    ticker_extraction,
    data_dir: str,
    default_year: int,
    document_dao: DocumentDao,
) -> Tuple[List[int], List[str], List[str], str]:
    """
    Resolve extracted ticker/year pairs to database document IDs.

    Tries exact-path lookup first (matching indexed document_path).
    Falls back to filename-based lookup if exact paths don't match.

    Returns:
        (document_ids, routed_file_paths, failed_tickers, routing_status)
    """
    file_paths: List[str] = []
    failed_tickers: List[str] = []

    for ticker_info in ticker_extraction.items:
        ticker = ticker_info.ticker_symbol
        years = ticker_info.years

        ticker_found = False
        for year in years:
            found = find_document_file(data_dir, year, ticker)
            if found:
                file_paths.append(os.path.abspath(found))
                ticker_found = True

        if not ticker_found:
            # Fallback to default year
            fallback = find_document_file(data_dir, default_year, ticker)
            if fallback:
                file_paths.append(os.path.abspath(fallback))
            else:
                failed_tickers.append(ticker)

    if not file_paths:
        return [], [], failed_tickers, "no_files_on_disk"

    # --- Primary: exact document_path match ---
    documents = document_dao.get_documents_by_paths(file_paths)
    if documents:
        doc_ids = [d.id for d in documents]
        matched_paths = [d.document_path for d in documents]
        return doc_ids, matched_paths, failed_tickers, "exact_path"

    # --- Fallback: match by file_name ---
    file_names = [os.path.basename(p) for p in file_paths]
    documents = document_dao.get_documents_by_file_names(file_names)
    if documents:
        doc_ids = [d.id for d in documents]
        matched_paths = [d.document_path for d in documents]
        logger.warning(
            f"Exact path match failed; fell back to filename match. "
            f"Requested: {file_names}, Matched: {matched_paths}"
        )
        return doc_ids, matched_paths, failed_tickers, "filename_fallback"

    return [], file_paths, failed_tickers, "not_in_index"


# ---------------------------------------------------------------------------
# Query processing
# ---------------------------------------------------------------------------


async def process_single_query(
    query_item: QueryItem,
    data_dir: str,
    default_year: int,
    rewriting_service: RewritingService,
    query_processor: QueryProcessor,
    inference_facade: InferenceFacade,
    document_dao: DocumentDao,
    progress_info: Optional[Tuple[int, int, int]] = None,
) -> QueryResult:
    """
    Process a single query through the hybrid SFR+CBR pipeline.

    1. Extract ticker/year (SFR stage)
    2. Resolve to document IDs in the index DB
    3. Run scoped hybrid retrieval (CBR stage)
    4. Generate answer from retrieved chunks
    """
    result = QueryResult(query_item=query_item)

    try:
        progress_str = ""
        if progress_info:
            cur, start, end = progress_info
            progress_str = f" [Progress: {cur - start + 1}/{end - start + 1}]"

        logger.info(f"Processing query [{query_item.query_id}]{progress_str}: {query_item.query[:100]}...")

        # ------ Stage 1: Document routing via ticker extraction ------
        ticker_extraction = await rewriting_service.extract_ticker_info_async(
            query_item.query, default_year
        )

        doc_ids = None  # None = unrestricted (full-database) search

        if ticker_extraction.is_empty():
            logger.warning(
                f"No ticker symbols extracted for query [{query_item.query_id}], "
                f"falling back to full-database retrieval"
            )
            result.metadata["routing_status"] = "fallback_no_tickers"
        else:
            result.metadata["tickers"] = ",".join(i.ticker_symbol for i in ticker_extraction.items)

            doc_ids_resolved, routed_paths, failed_tickers, routing_status = resolve_document_ids(
                ticker_extraction, data_dir, default_year, document_dao
            )
            result.metadata["routed_documents"] = ",".join(routed_paths)

            if failed_tickers:
                result.metadata["failed_tickers"] = ",".join(failed_tickers)

            if doc_ids_resolved:
                doc_ids = doc_ids_resolved
                result.metadata["routing_status"] = routing_status
                logger.info(
                    f"Query [{query_item.query_id}]: Routed to {len(doc_ids)} document(s) "
                    f"(status={routing_status}, ids={doc_ids})"
                )
            else:
                result.metadata["routing_status"] = f"fallback_{routing_status}"
                logger.warning(
                    f"Query [{query_item.query_id}]: Document routing failed ({routing_status}), "
                    f"falling back to full-database retrieval"
                )

        # ------ Stage 2: Chunk-based retrieval (scoped if routed, full-database otherwise) ------
        chat_context_units = await query_processor.process_query_async(
            query_item.query, document_ids=doc_ids
        )
        result.metadata["chunk_ids"] = ",".join(str(u.chunk_id) for u in chat_context_units)
        result.metadata["document_ids"] = ",".join(str(did) for did in set(u.document_id for u in chat_context_units))

        logger.info(f"Query [{query_item.query_id}]: Retrieved {len(chat_context_units)} chunks from scoped search")

        # ------ Stage 3: Answer generation ------
        if chat_context_units:
            context_str = "\n\n---\n\n".join(
                f"[Chunk {i+1}] {unit.chunk_source}:\n{unit.chunk_content}"
                for i, unit in enumerate(chat_context_units)
            )
            user_prompt = RAG_ANSWER_GENERATION_USER_PROMPT.format(
                context=context_str, query=query_item.query
            )
            messages = [
                {"role": "system", "content": RAG_ANSWER_GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            final_answer = await inference_facade.create_chat_completion_async(messages)
            result.final_answer = final_answer.strip()
            logger.info(f"Completed query [{query_item.query_id}]")
        else:
            result.final_answer = (
                "No relevant chunks found within the routed document(s). "
                "The document was identified but no matching content was retrieved."
            )
            logger.warning(f"No chunks retrieved for query [{query_item.query_id}]")

    except Exception as e:
        logger.error(f"Error processing query [{query_item.query_id}]: {e}")
        logger.debug(f"Error details for query [{query_item.query_id}]", exc_info=True)
        result.error = str(e)
        result.final_answer = f"Error: {e}"

    return result


async def process_queries_concurrently(
    query_items: List[QueryItem],
    data_dir: str,
    default_year: int,
    rewriting_service: RewritingService,
    query_processor: QueryProcessor,
    inference_facade: InferenceFacade,
    document_dao: DocumentDao,
    max_concurrent_tasks: int = 3,
    start_row: Optional[int] = None,
    end_row: Optional[int] = None,
    csv_writer: Optional[StreamingCSVWriter] = None,
) -> List[QueryResult]:
    """Process multiple queries concurrently through the hybrid pipeline."""
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    results_list: List[Optional[QueryResult]] = [None] * len(query_items)

    async def _process(qi: QueryItem, idx: int):
        async with semaphore:
            progress_info = None
            original_row = idx + 1
            if start_row is not None and end_row is not None:
                original_row = start_row + idx
                progress_info = (original_row, start_row, end_row)

            try:
                r = await process_single_query(
                    qi, data_dir, default_year, rewriting_service,
                    query_processor, inference_facade, document_dao,
                    progress_info,
                )
            except Exception as e:
                logger.error(f"Exception in query processing: {e}")
                r = QueryResult(query_item=qi, error=str(e), final_answer=f"Error: {e}")

            results_list[idx] = r
            if csv_writer:
                await csv_writer.write_result(r, original_row)

    tasks = [_process(qi, i) for i, qi in enumerate(query_items)]
    await asyncio.gather(*tasks, return_exceptions=False)
    return [r for r in results_list if r is not None]


# ---------------------------------------------------------------------------
# Service initialization
# ---------------------------------------------------------------------------


def get_project_paths(project_name: Optional[str] = None) -> Tuple[str, str]:
    if project_name:
        return f"{project_name}.rag_pipeline.db", f"{project_name}.faiss_index.index"
    return DEFAULT_DB_NAME, DEFAULT_FAISS_INDEX_NAME


def initialize_services(
    db_path: str,
    faiss_index_path: str,
    use_local_reranker: bool = False,
    local_reranker_num_workers: int = DEFAULT_LOCAL_RERANKER_NUM_WORKERS,
):
    """Initialize all required services (DB, FAISS, embedder, reranker)."""
    logger.info("Initializing services...")
    logger.info(f"Database: {db_path}")
    logger.info(f"FAISS Index: {faiss_index_path}")

    schema_path = str(project_root / "schemas" / "fts_schema.sql")
    db_manager = Sqlite3DbManager(db_path=db_path, schema_file=schema_path)
    Session = db_manager.session_factory

    document_dao = DocumentDao(Session)
    chunk_dao = ChunkDao(Session)

    embedder = Embedder(EmbedderConfig())

    faiss_index = FaissIndex(
        index_file=faiss_index_path,
        dimension=DEFAULT_EMBEDDING_DIM,
        faiss_lock=None,
    )
    faiss_index.load_or_build_from_scratch()

    vector_search_service = VectorSearchService(
        embedder=embedder,
        faiss_index=faiss_index,
        default_limit=DEFAULT_SEMANTIC_RETRIEVAL_LIMIT,
    )

    if use_local_reranker:
        logger.info("Using LOCAL reranker (nexaai)")
        repo_root = project_root.parent
        model_path = str(repo_root / DEFAULT_LOCAL_RERANKER_MODEL_PATH)
        tokenizer_path = str(repo_root / DEFAULT_LOCAL_RERANKER_TOKENIZER_PATH)
        reranker = LocalRerankService(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            num_workers=local_reranker_num_workers,
        )
    else:
        logger.info("Using CLOUD reranker (Jina API)")
        reranker = JinaRerankService()

    logger.info("Services initialized successfully")
    return {
        "document_dao": document_dao,
        "chunk_dao": chunk_dao,
        "embedder": embedder,
        "faiss_index": faiss_index,
        "vector_search_service": vector_search_service,
        "reranker": reranker,
    }


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Hybrid SFR+CBR Query Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single query
  python main/main_hybrid_query.py --query "What was AAPL's revenue?" --data-dir ./reports

  # With custom project name (uses myproject.rag_pipeline.db and myproject.faiss_index.index)
  python main/main_hybrid_query.py --query "What was AAPL's revenue?" --data-dir ./reports -p myproject

  # Queries from CSV
  python main/main_hybrid_query.py --csv queries.csv --column question --data-dir ./reports --output-csv

  # Process specific row range
  python main/main_hybrid_query.py --csv queries.csv --column question --data-dir ./reports --start-row 1 --end-row 300

Pipeline Architecture:
  Stage 1 (SFR):  Extract ticker/year -> resolve to indexed document IDs
  Stage 2 (CBR):  Hybrid search (FTS + semantic + RRF + reranking) scoped to routed documents
  Stage 3:        Answer generation from targeted chunks

  This combines SFR's robustness (no cross-document confusion) with CBR's
  precision (targeted chunk retrieval), eliminating the trade-off between the two.
        """,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--query", type=str, help="Single query from command line")
    input_group.add_argument("--csv", type=str, metavar="PATH", help="Path to CSV file containing queries")
    input_group.add_argument("--hf-dataset", type=str, metavar="DATASET_ID", help="HuggingFace dataset ID")

    parser.add_argument("--data-dir", type=str, required=True, metavar="PATH",
                        help="Path to directory containing financial reports organized as {year}/{ticker}.{ext}")
    parser.add_argument("--default-year", type=int, default=DEFAULT_YEAR, metavar="YEAR",
                        help=f"Default year when query omits year (default: {DEFAULT_YEAR})")
    parser.add_argument("-p", "--project-name", type=str, default=None, metavar="NAME",
                        help="Project name for database and index files")
    parser.add_argument("--column", type=str, metavar="NAME", help="Column name containing queries")
    parser.add_argument("--id-column", type=str, metavar="NAME", help="Column name for query IDs")
    parser.add_argument("--split", type=str, default="train", metavar="NAME",
                        help="Dataset split for HuggingFace datasets (default: train)")
    parser.add_argument("--start-row", type=int, default=None, metavar="N",
                        help="Starting row number (1-based, inclusive)")
    parser.add_argument("--end-row", type=int, default=None, metavar="N",
                        help="Ending row number (1-based, inclusive)")
    parser.add_argument("--output-csv", type=str, nargs="?", const="", default=None, metavar="PATH",
                        help="Output to CSV (auto-generates filename if flag given without path)")
    parser.add_argument("--concurrent", type=int, default=3, metavar="N",
                        help="Number of concurrent query tasks (default: 3)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, metavar="NAME",
                        help=f"Model for answer generation (default: {DEFAULT_MODEL})")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL, metavar="URL",
                        help=f"Base URL for inference API (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--no-rewrite", action="store_true", help="Disable query rewriting for chunk retrieval")
    parser.add_argument("--no-rerank", action="store_true", help="Disable reranking")
    parser.add_argument("--rerank-use-local", action="store_true",
                        help="Use local reranker (nexaai) instead of Jina API")
    parser.add_argument("--rerank-local-workers", type=int, default=DEFAULT_LOCAL_RERANKER_NUM_WORKERS, metavar="N",
                        help=f"Worker processes for local reranker (default: {DEFAULT_LOCAL_RERANKER_NUM_WORKERS})")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main():
    args = parse_arguments()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("=" * 80)
    print("HYBRID SFR+CBR QUERY PIPELINE")
    print("=" * 80)

    # Validate data directory
    if not os.path.isdir(args.data_dir):
        logger.error(f"Data directory does not exist or is not a directory: {args.data_dir}")
        sys.exit(1)

    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Default year: {args.default_year}")

    # Database / index paths
    db_path, faiss_index_path = get_project_paths(args.project_name)
    if args.project_name:
        logger.info(f"Project: {args.project_name}  DB: {db_path}  FAISS: {faiss_index_path}")

    # Validate column argument
    if (args.csv or args.hf_dataset) and not args.column:
        logger.error("--column is required when using --csv or --hf-dataset")
        sys.exit(1)

    # Load queries
    try:
        input_source = create_input_source(args)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    try:
        query_items = await input_source.get_queries()
    except Exception as e:
        logger.error(f"Failed to load queries: {e}")
        sys.exit(1)

    if not query_items:
        logger.warning("No queries found in input source")
        return

    total_loaded = len(query_items)
    logger.info(f"Loaded {total_loaded} queries")

    # Row-range filtering
    start_row = args.start_row
    end_row = args.end_row

    if args.query:
        if start_row is not None or end_row is not None:
            logger.warning("--start-row/--end-row ignored for CLI queries")
        start_row, end_row = 1, 1
    else:
        if start_row is None:
            start_row = 1
        if end_row is None:
            end_row = total_loaded
        if start_row < 1:
            logger.error(f"--start-row must be >= 1, got {start_row}")
            sys.exit(1)
        if end_row < start_row:
            logger.error(f"--end-row ({end_row}) must be >= --start-row ({start_row})")
            sys.exit(1)
        if start_row > total_loaded:
            logger.error(f"--start-row ({start_row}) exceeds total rows ({total_loaded})")
            sys.exit(1)
        if end_row > total_loaded:
            logger.warning(f"--end-row clamped from {end_row} to {total_loaded}")
            end_row = total_loaded
        query_items = query_items[start_row - 1 : end_row]
        logger.info(f"Filtered to rows {start_row}-{end_row}: {len(query_items)} queries")

    # Initialize services
    services = initialize_services(
        db_path, faiss_index_path,
        use_local_reranker=args.rerank_use_local,
        local_reranker_num_workers=args.rerank_local_workers,
    )

    inference_facade = InferenceFacade(model=args.model, base_url=args.base_url)

    # Rewriting service (used for both ticker extraction and optional query rewrite)
    rewriting_service = RewritingService(inference_facade)

    # Build QueryProcessor for the CBR stage
    retriever_config = RetrieverConfig(
        semantic_filtering_threshold=DEFAULT_SEMANTIC_FILTERING_THRESHOLD,
        final_chunks_to_assemble_limit=DEFAULT_FINAL_CHUNKS_TO_ASSEMBLE_LIMIT,
        fts_retrieval_limit=DEFAULT_FTS_RETRIEVAL_LIMIT,
        semantic_retrieval_limit=DEFAULT_SEMANTIC_RETRIEVAL_LIMIT,
        reciprocal_rank_fusion_k=DEFAULT_RRF_K,
    )
    query_config = QueryConfig(
        retriever_config=retriever_config,
        rerank_keep_threshold=DEFAULT_RERANK_KEEP_THRESHOLD,
        rerank_cliff_cutoff_score_difference=DEFAULT_RERANK_CLIFF_CUTOFF_SCORE_DIFFERENCE,
        run_rewrite=not args.no_rewrite,
        run_retrieve=True,
        run_rerank=not args.no_rerank,
        run_with_keyword_extraction=not args.no_rewrite,
        max_item_to_rerank=DEFAULT_MAX_ITEM_TO_RERANK,
    )
    query_processor = QueryProcessor(
        config=query_config,
        chunk_dao=services["chunk_dao"],
        document_dao=services["document_dao"],
        faiss_index=services["faiss_index"],
        embedder=services["embedder"],
        reranker=services["reranker"] if query_config.run_rerank else None,
        inference_service=inference_facade if query_config.run_rewrite else None,
        rewriting_service=rewriting_service if query_config.run_rewrite else None,
        vector_search_service=services["vector_search_service"],
    )

    # Output CSV setup
    output_csv_path = None
    if args.output_csv is not None:
        if args.output_csv == "":
            source = Path(args.csv).stem if args.csv else (args.hf_dataset or "cli").replace("/", "_")
            model_tag = args.model.replace("/", "_").replace(":", "_")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("./output")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_csv_path = str(output_dir / f"hybrid-{source}-{start_row}-to-{end_row}-{model_tag}-{ts}.csv")
            logger.info(f"Auto-generated output path: {output_csv_path}")
        else:
            output_csv_path = args.output_csv

    csv_writer = None
    fieldnames = [
        "original_row", "query_id", "query", "tickers", "routed_documents",
        "routing_status", "chunk_ids", "document_ids", "final_answer", "error",
    ]
    if output_csv_path:
        csv_writer = StreamingCSVWriter(output_csv_path, fieldnames)
        await csv_writer.initialize()
        logger.info(f"Streaming results to: {output_csv_path}")

    # Process queries
    print(f"\nProcessing {len(query_items)} queries (rows {start_row} to {end_row})...")

    results = await process_queries_concurrently(
        query_items, args.data_dir, args.default_year,
        rewriting_service, query_processor, inference_facade,
        services["document_dao"],
        max_concurrent_tasks=args.concurrent,
        start_row=start_row, end_row=end_row,
        csv_writer=csv_writer,
    )

    if not output_csv_path:
        output_to_terminal(results)

    # Summary
    successful = sum(1 for r in results if not r.error)
    failed = len(results) - successful
    routing_statuses = {}
    for r in results:
        s = r.metadata.get("routing_status", "unknown")
        routing_statuses[s] = routing_statuses.get(s, 0) + 1

    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Row range:          {start_row} to {end_row}")
    print(f"Total queries:      {len(results)}")
    print(f"Successful:         {successful}")
    print(f"Failed:             {failed}")
    print(f"Routing breakdown:  {routing_statuses}")
    if output_csv_path:
        print(f"Output file:        {output_csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
