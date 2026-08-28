#!/usr/bin/env python
"""
Agentic RAG query pipeline (Phase 2 / Experiment 3 baseline).

Self-correcting retrieval agent that does *post-hoc* entity verification
instead of upfront routing. The contrast with HDRR (which routes BEFORE
retrieval) is exactly what the §7.2 cost-vs-correctness comparison in
the ISWA submission rests on.

Pipeline:
  Stage 1: Extract ticker from query                          (1 LLM call)
  Stage 2: Full-corpus CBR retrieval                          (no scoping)
  Stage 3: Entity verification call                           (1 LLM call)
           "Do these chunks belong to {ticker}? YES/NO + reason"
  Stage 4: If NO -> reformulate query with entity anchor,
           re-run full-corpus CBR + verification (up to 2 retries)
  Stage 5: Answer generation                                  (1 LLM call)

If ticker extraction returns nothing, the agent skips verification and
proceeds straight to full-corpus CBR + answer generation - the same
fallback policy HDRR uses for unannotated queries, kept here so the
two systems are compared on equal footing.

This script reuses every existing artefact (DB, FAISS, embedder,
reranker). The only new pieces are the verification prompts and the
retry loop.
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
import time
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
from data_classes.inference_parameters import InferenceParameters
from data_classes.query_config import QueryConfig
from data_classes.retriever_config import RetrieverConfig
from main.query_io import (
    QueryItem,
    QueryResult,
    create_input_source,
    output_to_terminal,
)
from prompts import (
    RAG_ANSWER_GENERATION_SYSTEM_PROMPT,
    RAG_ANSWER_GENERATION_USER_PROMPT,
    AGENTIC_VERIFICATION_SYSTEM_PROMPT,
    AGENTIC_VERIFICATION_USER_PROMPT,
)
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
DEFAULT_MAX_RETRIES = 2
VERIFICATION_CHUNK_PREVIEW_COUNT = 3
VERIFICATION_CHUNK_PREVIEW_CHARS = 300


# ---------------------------------------------------------------------------
# Streaming CSV writer
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
            "‘": "'", "’": "'", "“": '"', "”": '"',
            "–": "-", "—": "--", "×": "x", "÷": "/",
            "−": "-", "±": "+/-", "…": "...", "•": "-",
            "°": " deg", " ": " ",
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
                "verification_result": self._sanitize_text(result.metadata.get("verification_result", "")),
                "verification_reasons": self._sanitize_text(result.metadata.get("verification_reasons", "")),
                "retry_count": self._sanitize_text(result.metadata.get("retry_count", "")),
                "total_llm_calls": self._sanitize_text(result.metadata.get("total_llm_calls", "")),
                "elapsed_ms": self._sanitize_text(result.metadata.get("elapsed_ms", "")),
                "chunk_ids": self._sanitize_text(result.metadata.get("chunk_ids", "")),
                "document_ids": self._sanitize_text(result.metadata.get("document_ids", "")),
                "final_answer": self._sanitize_text(result.final_answer),
                "error": self._sanitize_text(result.error),
            }
            row = {k: v for k, v in row.items() if k in self.fieldnames}
            with open(self.filepath, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames, quoting=csv.QUOTE_ALL)
                writer.writerow(row)


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------


def build_chunks_preview(chat_context_units) -> str:
    """Format the top-K retrieved chunks for the verification prompt."""
    lines = []
    for i, unit in enumerate(chat_context_units[:VERIFICATION_CHUNK_PREVIEW_COUNT]):
        snippet = (unit.chunk_content or "").strip().replace("\n", " ")
        if len(snippet) > VERIFICATION_CHUNK_PREVIEW_CHARS:
            snippet = snippet[:VERIFICATION_CHUNK_PREVIEW_CHARS] + "..."
        lines.append(f"[Chunk {i+1}] source={unit.chunk_source}\n{snippet}")
    return "\n\n".join(lines) if lines else "(no chunks retrieved)"


async def verify_chunks_for_ticker(
    inference_facade: InferenceFacade,
    query: str,
    ticker: str,
    chat_context_units,
) -> Tuple[bool, str]:
    """Single LLM verification call. Returns (belongs, reason)."""
    chunks_preview = build_chunks_preview(chat_context_units)
    messages = [
        {"role": "system", "content": AGENTIC_VERIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": AGENTIC_VERIFICATION_USER_PROMPT.format(
            query=query, ticker=ticker, chunks_preview=chunks_preview,
        )},
    ]
    response = await inference_facade.create_chat_completion_async(
        messages,
        inference_parameters=InferenceParameters(
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=200,
        ),
    )
    try:
        parsed = json.loads(response)
        belongs = bool(parsed.get("belongs", False))
        reason = str(parsed.get("reason", "")).strip()
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logger.warning(f"Verification JSON parse failed: {e}; response={response[:200]!r}")
        # Conservative default: assume YES so we don't trigger a wasteful retry
        # on bad JSON. This biases agentic toward the cheaper path.
        belongs, reason = True, f"parse_error: {e}"
    return belongs, reason


def anchor_query_with_ticker(original_query: str, ticker: str) -> str:
    """Reformulate a query by prepending an entity anchor."""
    return f"In {ticker}'s 10-K annual report: {original_query}"


# ---------------------------------------------------------------------------
# Query processing
# ---------------------------------------------------------------------------


async def process_single_query(
    query_item: QueryItem,
    default_year: int,
    rewriting_service: RewritingService,
    query_processor: QueryProcessor,
    inference_facade: InferenceFacade,
    max_retries: int,
    progress_info: Optional[Tuple[int, int, int]] = None,
) -> QueryResult:
    """Process one query through the agentic verification + retry loop."""
    result = QueryResult(query_item=query_item)
    start_time = time.perf_counter()
    llm_calls = 0
    retry_count = 0
    verification_history: List[str] = []
    reasons_history: List[str] = []

    try:
        progress_str = ""
        if progress_info:
            cur, start, end = progress_info
            progress_str = f" [Progress: {cur - start + 1}/{end - start + 1}]"
        logger.info(f"Processing query [{query_item.query_id}]{progress_str}: {query_item.query[:100]}...")

        # ------ Stage 1: ticker extraction (1 LLM call) ------
        ticker_extraction = await rewriting_service.extract_ticker_info_async(
            query_item.query, default_year
        )
        llm_calls += 1

        target_ticker: Optional[str] = None
        if not ticker_extraction.is_empty():
            target_ticker = ticker_extraction.items[0].ticker_symbol
            result.metadata["tickers"] = ",".join(i.ticker_symbol for i in ticker_extraction.items)

        # ------ Stage 2: first-pass full-corpus retrieval ------
        current_query = query_item.query
        chat_context_units = await query_processor.process_query_async(
            current_query, document_ids=None
        )
        logger.info(f"Query [{query_item.query_id}]: First-pass retrieved {len(chat_context_units)} chunks (full corpus)")

        # ------ Stage 3+4: verification + retry (only if a ticker was extracted) ------
        if target_ticker is not None:
            for attempt in range(max_retries + 1):
                belongs, reason = await verify_chunks_for_ticker(
                    inference_facade, current_query, target_ticker, chat_context_units
                )
                llm_calls += 1
                verification_history.append("YES" if belongs else "NO")
                reasons_history.append(reason)

                if belongs:
                    break  # accept retrieval, move on to answer generation

                if attempt >= max_retries:
                    logger.info(
                        f"Query [{query_item.query_id}]: Verification still NO after "
                        f"{max_retries} retries; proceeding with last retrieval."
                    )
                    break

                retry_count += 1
                current_query = anchor_query_with_ticker(query_item.query, target_ticker)
                logger.info(
                    f"Query [{query_item.query_id}]: Retry {retry_count}/{max_retries} "
                    f"with anchored query: {current_query[:120]}..."
                )
                chat_context_units = await query_processor.process_query_async(
                    current_query, document_ids=None
                )
        else:
            # No ticker -> skip verification (mirrors HDRR's fallback)
            verification_history.append("SKIPPED")
            reasons_history.append("no ticker extracted")

        result.metadata["verification_result"] = "|".join(verification_history)
        result.metadata["verification_reasons"] = "|".join(reasons_history)
        result.metadata["retry_count"] = str(retry_count)
        result.metadata["chunk_ids"] = ",".join(str(u.chunk_id) for u in chat_context_units)
        result.metadata["document_ids"] = ",".join(
            str(did) for did in set(u.document_id for u in chat_context_units)
        )

        # ------ Stage 5: answer generation (1 LLM call) ------
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
            llm_calls += 1
            result.final_answer = final_answer.strip()
            logger.info(f"Completed query [{query_item.query_id}]")
        else:
            result.final_answer = (
                "No relevant chunks found within the corpus. "
                "The retrieval pipeline returned no candidates."
            )
            logger.warning(f"No chunks retrieved for query [{query_item.query_id}]")

    except Exception as e:
        logger.error(f"Error processing query [{query_item.query_id}]: {e}")
        logger.debug(f"Error details for query [{query_item.query_id}]", exc_info=True)
        result.error = str(e)
        result.final_answer = f"Error: {e}"

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)
    result.metadata["elapsed_ms"] = str(elapsed_ms)
    result.metadata["total_llm_calls"] = str(llm_calls)
    return result


async def process_queries_concurrently(
    query_items: List[QueryItem],
    default_year: int,
    rewriting_service: RewritingService,
    query_processor: QueryProcessor,
    inference_facade: InferenceFacade,
    max_retries: int = DEFAULT_MAX_RETRIES,
    max_concurrent_tasks: int = 3,
    start_row: Optional[int] = None,
    end_row: Optional[int] = None,
    csv_writer: Optional[StreamingCSVWriter] = None,
) -> List[QueryResult]:
    """Process multiple queries concurrently through the agentic pipeline."""
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
                    qi, default_year, rewriting_service, query_processor,
                    inference_facade, max_retries, progress_info,
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
    """Initialize DB, FAISS, embedder, reranker."""
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
        description="Agentic RAG Query Pipeline (verification + retry baseline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main/main_agentic_query.py --query "What was AAPL's revenue?" -p full_10k
  python main/main_agentic_query.py --csv queries.csv --column question -p full_10k --output-csv
  python main/main_agentic_query.py --hf-dataset Linq-AI-Research/FinDER --column query -p full_10k --output-csv
        """,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--query", type=str, help="Single query from command line")
    input_group.add_argument("--csv", type=str, metavar="PATH", help="Path to CSV file containing queries")
    input_group.add_argument("--hf-dataset", type=str, metavar="DATASET_ID", help="HuggingFace dataset ID")

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
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, metavar="N",
                        help=f"Max verification retries (default: {DEFAULT_MAX_RETRIES})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, metavar="NAME",
                        help=f"Model for verification and answer generation (default: {DEFAULT_MODEL})")
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
    print("AGENTIC RAG QUERY PIPELINE")
    print("=" * 80)

    db_path, faiss_index_path = get_project_paths(args.project_name)
    if args.project_name:
        logger.info(f"Project: {args.project_name}  DB: {db_path}  FAISS: {faiss_index_path}")

    if (args.csv or args.hf_dataset) and not args.column:
        logger.error("--column is required when using --csv or --hf-dataset")
        sys.exit(1)

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

    services = initialize_services(
        db_path, faiss_index_path,
        use_local_reranker=args.rerank_use_local,
        local_reranker_num_workers=args.rerank_local_workers,
    )

    inference_facade = InferenceFacade(model=args.model, base_url=args.base_url)
    rewriting_service = RewritingService(inference_facade)

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

    output_csv_path = None
    if args.output_csv is not None:
        if args.output_csv == "":
            source = Path(args.csv).stem if args.csv else (args.hf_dataset or "cli").replace("/", "_")
            model_tag = args.model.replace("/", "_").replace(":", "_")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = project_root.parent / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_csv_path = str(output_dir / f"agentic-{source}-{start_row}-to-{end_row}-{model_tag}-{ts}.csv")
            logger.info(f"Auto-generated output path: {output_csv_path}")
        else:
            output_csv_path = args.output_csv

    csv_writer = None
    fieldnames = [
        "original_row", "query_id", "query", "tickers",
        "verification_result", "verification_reasons", "retry_count",
        "total_llm_calls", "elapsed_ms",
        "chunk_ids", "document_ids", "final_answer", "error",
    ]
    if output_csv_path:
        csv_writer = StreamingCSVWriter(output_csv_path, fieldnames)
        await csv_writer.initialize()
        logger.info(f"Streaming results to: {output_csv_path}")

    print(f"\nProcessing {len(query_items)} queries (rows {start_row} to {end_row})...")
    results = await process_queries_concurrently(
        query_items, args.default_year,
        rewriting_service, query_processor, inference_facade,
        max_retries=args.max_retries,
        max_concurrent_tasks=args.concurrent,
        start_row=start_row, end_row=end_row,
        csv_writer=csv_writer,
    )

    if not output_csv_path:
        output_to_terminal(results)

    successful = sum(1 for r in results if not r.error)
    failed = len(results) - successful
    retries = [int(r.metadata.get("retry_count", "0") or "0") for r in results]
    total_retries = sum(retries)
    n_retried = sum(1 for r in retries if r > 0)
    avg_llm_calls = (
        sum(int(r.metadata.get("total_llm_calls", "0") or "0") for r in results)
        / max(len(results), 1)
    )

    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Row range:            {start_row} to {end_row}")
    print(f"Total queries:        {len(results)}")
    print(f"Successful:           {successful}")
    print(f"Failed:               {failed}")
    print(f"Queries with retry:   {n_retried} ({n_retried/max(len(results),1)*100:.1f}%)")
    print(f"Total retries issued: {total_retries}")
    print(f"Avg LLM calls/query:  {avg_llm_calls:.2f}")
    if output_csv_path:
        print(f"Output file:          {output_csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
