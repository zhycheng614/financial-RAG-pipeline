#!/usr/bin/env python
"""
Contextual Retrieval re-indexer (Phase 1 / Experiment 1).

Builds a new SQLite + FAISS index from an existing source index by
prepending document-level context to each chunk before embedding.
Two variants are supported:

  metadata  Static "[Company: {TICKER}, Fiscal Year: {YEAR}] " prefix.
            Zero LLM cost. Isolates the contribution of pure identity tagging.

  llm       Anthropic 2024-style per-chunk generated context. For each chunk,
            the LLM sees a short preview of the parent document plus the
            chunk text and returns a 50-100 token situating paragraph,
            which is prepended to the chunk before embedding.

The source index is read-only - we never modify `full_10k.*`. The output
is a fresh project (`contextual_meta` or `contextual_llm`) with its own
.rag_pipeline.db and .faiss_index.index.

Examples:
  # Variant A (cheap, fast)
  python main/main_index_contextual.py \\
      --source-project full_10k --contextual metadata \\
      --output-project contextual_meta --default-year 2023

  # Variant B (LLM-generated, pilot one document first)
  python main/main_index_contextual.py \\
      --source-project full_10k --contextual llm \\
      --output-project contextual_llm_pilot \\
      --document-limit 1
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from beans.embedder import Embedder
from beans.faiss_manager import FaissIndex
from beans.sqlite3_db_manager import Sqlite3DbManager
from beans.inference_facade import InferenceFacade
from dao.document_dao import DocumentDao
from dao.chunk_dao import ChunkDao
from data_classes.embedding_generator_config import EmbedderConfig
from data_classes.inference_parameters import InferenceParameters
from models.document import Document
from models.chunk import Chunk
from constants import DEFAULT_EMBEDDING_DIM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CONTEXT_GEN_SYSTEM_PROMPT = """\
You are writing short, factual "situating" paragraphs that help a retrieval
system disambiguate chunks of 10-K annual reports across many companies.

Given a small preview of the source document plus one chunk from that
document, write a 50-100 token paragraph that:
- States the company name and ticker if identifiable from the preview.
- States the fiscal year if identifiable, otherwise the most likely year.
- States the section or topic the chunk belongs to.
- States the chunk's role in 1 short sentence.

Output the situating paragraph as plain text. No JSON, no markdown, no
quotes, no labels - just the paragraph itself.
"""

CONTEXT_GEN_USER_PROMPT = """\
SOURCE DOCUMENT PREVIEW (first portion of {file_name}):
{doc_preview}

CHUNK TO SITUATE:
{chunk_text}

Write the 50-100 token situating paragraph now."""

DEFAULT_SOURCE_PROJECT = "full_10k"
DEFAULT_LLM_MODEL = "gpt-4.1-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
# Hard cap on the doc preview length (in characters). Caching saves us only
# on the preview portion, so this number directly drives cost.
DEFAULT_DOC_PREVIEW_CHARS = 6000  # ~1500 tokens
DEFAULT_CONTEXT_MAX_TOKENS = 120


def project_paths(name: str) -> Tuple[str, str]:
    return f"{name}.rag_pipeline.db", f"{name}.faiss_index.index"


def ticker_from_file_name(file_name: str) -> str:
    """`AAPL.pdf` -> `AAPL`. Fallback to stem if extension differs."""
    return Path(file_name).stem


def build_metadata_prefix(ticker: str, year: int) -> str:
    return f"[Company: {ticker}, Fiscal Year: {year}]\n---\n"


def build_doc_preview(chunks: List[Chunk], max_chars: int) -> str:
    """Concatenate the first N chunks (in id order) up to a char cap."""
    parts: List[str] = []
    total = 0
    for c in chunks:
        text = (c.chunk_text or "").strip()
        if not text:
            continue
        if total + len(text) > max_chars:
            parts.append(text[: max(0, max_chars - total)])
            break
        parts.append(text)
        total += len(text)
        if total >= max_chars:
            break
    return "\n\n".join(parts)


async def generate_context_for_chunk(
    inference_facade: InferenceFacade,
    file_name: str,
    doc_preview: str,
    chunk_text: str,
    max_tokens: int,
) -> str:
    user_prompt = CONTEXT_GEN_USER_PROMPT.format(
        file_name=file_name,
        doc_preview=doc_preview,
        chunk_text=chunk_text,
    )
    messages = [
        {"role": "system", "content": CONTEXT_GEN_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    response = await inference_facade.create_chat_completion_async(
        messages,
        inference_parameters=InferenceParameters(
            temperature=0.0,
            max_tokens=max_tokens,
        ),
    )
    return response.strip()


async def process_document_llm(
    document: Document,
    chunks: List[Chunk],
    inference_facade: InferenceFacade,
    doc_preview_chars: int,
    context_max_tokens: int,
    concurrent: int,
) -> Tuple[List[str], Dict[str, float]]:
    """Generate per-chunk contexts for one document; returns prefixed texts."""
    doc_preview = build_doc_preview(chunks, doc_preview_chars)
    semaphore = asyncio.Semaphore(concurrent)
    contexts: List[Optional[str]] = [None] * len(chunks)
    t0 = time.perf_counter()

    async def _gen(i: int, ch: Chunk):
        async with semaphore:
            ctx = await generate_context_for_chunk(
                inference_facade,
                document.file_name,
                doc_preview,
                ch.chunk_text or "",
                context_max_tokens,
            )
            contexts[i] = ctx

    await asyncio.gather(*(_gen(i, ch) for i, ch in enumerate(chunks)))

    elapsed = time.perf_counter() - t0
    prefixed_texts = [
        f"{contexts[i]}\n---\n{chunks[i].chunk_text or ''}"
        for i in range(len(chunks))
    ]
    stats = {
        "elapsed_s": elapsed,
        "chunks": len(chunks),
        "chunks_per_s": (len(chunks) / elapsed) if elapsed > 0 else 0.0,
        "doc_preview_chars": len(doc_preview),
    }
    return prefixed_texts, stats


async def reindex(
    args: argparse.Namespace,
) -> None:
    # ---- source DB (read-only intent) ----
    src_db_path, _ = project_paths(args.source_project)
    if not os.path.exists(src_db_path):
        logger.error(f"Source DB not found: {src_db_path}")
        sys.exit(1)
    schema_path = str(project_root / "schemas" / "fts_schema.sql")

    logger.info(f"Source DB: {src_db_path}")
    src_db = Sqlite3DbManager(db_path=src_db_path, schema_file=schema_path)
    src_doc_dao = DocumentDao(src_db.session_factory)
    src_chunk_dao = ChunkDao(src_db.session_factory)

    # ---- target DB ----
    tgt_db_path, tgt_faiss_path = project_paths(args.output_project)
    if os.path.exists(tgt_db_path) and not (args.overwrite or args.resume or args.build_faiss_only):
        logger.error(
            f"Target DB already exists: {tgt_db_path}. Use --overwrite, --resume, or --build-faiss-only."
        )
        sys.exit(1)
    if args.overwrite:
        for p in (tgt_db_path, tgt_faiss_path):
            if os.path.exists(p):
                logger.info(f"Removing existing: {p}")
                os.remove(p)
    if args.build_faiss_only and os.path.exists(tgt_faiss_path):
        # Stale empty FAISS file is fine to keep; the build step rewrites it.
        sz = os.path.getsize(tgt_faiss_path)
        if sz < 1000:
            logger.info(f"Removing stale empty FAISS file ({sz} bytes): {tgt_faiss_path}")
            os.remove(tgt_faiss_path)

    logger.info(f"Target DB: {tgt_db_path}")
    logger.info(f"Target FAISS: {tgt_faiss_path}")
    tgt_db = Sqlite3DbManager(db_path=tgt_db_path, schema_file=schema_path)
    tgt_doc_dao = DocumentDao(tgt_db.session_factory)
    tgt_chunk_dao = ChunkDao(tgt_db.session_factory)

    # ---- enumerate source documents ----
    with src_db.session_factory() as session:
        all_docs = session.query(Document).order_by(Document.id).all()
        all_docs = [
            Document(
                id=d.id,
                document_path=d.document_path,
                file_name=d.file_name,
                file_size=d.file_size,
                file_author=d.file_author,
                file_type=d.file_type,
            )
            for d in all_docs
        ]
    if args.document_limit is not None:
        all_docs = all_docs[: args.document_limit]
    if args.document_tickers:
        wanted = {t.strip().upper() for t in args.document_tickers.split(",") if t.strip()}
        all_docs = [d for d in all_docs if ticker_from_file_name(d.file_name).upper() in wanted]

    # --build-faiss-only short-circuits processing entirely.
    if args.build_faiss_only:
        logger.info("--build-faiss-only set; skipping all per-document processing.")
        all_docs = []

    # --resume skips documents whose file_name already exists in the target DB.
    if args.resume:
        with tgt_db.session_factory() as session:
            already_done = {
                d.file_name for d in session.query(Document).all() if d.file_name
            }
        before = len(all_docs)
        all_docs = [d for d in all_docs if d.file_name not in already_done]
        logger.info(
            f"--resume: {len(already_done)} doc(s) already in target DB; "
            f"skipping those, processing {len(all_docs)} remaining (filtered out {before - len(all_docs)})"
        )

    logger.info(f"Processing {len(all_docs)} document(s)")

    # ---- LLM facade for variant llm ----
    inference_facade: Optional[InferenceFacade] = None
    if args.contextual == "llm":
        inference_facade = InferenceFacade(model=args.llm_model, base_url=args.base_url)

    # ---- per-document loop ----
    total_chunks_written = 0
    t_global = time.perf_counter()
    per_doc_stats: List[Dict] = []

    for doc_idx, src_doc in enumerate(all_docs, 1):
        t_doc = time.perf_counter()
        ticker = ticker_from_file_name(src_doc.file_name)
        with src_db.session_factory() as session:
            src_chunks = (
                session.query(Chunk)
                .filter(Chunk.document_id == src_doc.id)
                .order_by(Chunk.id)
                .all()
            )
            src_chunks = [
                Chunk(
                    id=c.id,
                    document_id=c.document_id,
                    chunk_text=c.chunk_text,
                    file=c.file,
                    page=c.page,
                )
                for c in src_chunks
            ]

        logger.info(
            f"[{doc_idx}/{len(all_docs)}] doc_id={src_doc.id} ticker={ticker} "
            f"chunks={len(src_chunks)} file={src_doc.file_name}"
        )

        if args.contextual == "metadata":
            prefix = build_metadata_prefix(ticker, args.default_year)
            new_texts = [f"{prefix}{c.chunk_text or ''}" for c in src_chunks]
            stats = {"mode": "metadata"}
        else:
            new_texts, gen_stats = await process_document_llm(
                src_doc,
                src_chunks,
                inference_facade,
                doc_preview_chars=args.doc_preview_chars,
                context_max_tokens=args.context_max_tokens,
                concurrent=args.llm_concurrent,
            )
            stats = {"mode": "llm", **gen_stats}

        # Write new document + new chunks to target DB
        new_doc = Document(
            document_path=src_doc.document_path,
            file_name=src_doc.file_name,
            file_size=src_doc.file_size,
            file_author=src_doc.file_author,
            file_type=src_doc.file_type,
        )
        new_doc = tgt_doc_dao.add_return_obj(new_doc)
        new_chunks = [
            Chunk(
                document_id=new_doc.id,
                chunk_text=new_texts[i],
                file=src_chunks[i].file,
                page=src_chunks[i].page,
            )
            for i in range(len(src_chunks))
        ]
        if new_chunks:
            tgt_chunk_dao.add_all(new_chunks)
        total_chunks_written += len(new_chunks)
        stats.update(
            {
                "doc_id": src_doc.id,
                "ticker": ticker,
                "chunks": len(new_chunks),
                "wall_clock_s": time.perf_counter() - t_doc,
            }
        )
        per_doc_stats.append(stats)
        logger.info(
            f"  -> wrote {len(new_chunks)} chunks "
            f"(elapsed_doc={stats['wall_clock_s']:.1f}s)"
        )

    chunks_total_elapsed = time.perf_counter() - t_global
    logger.info(
        f"All documents processed: {total_chunks_written} chunks written in "
        f"{chunks_total_elapsed:.1f}s"
    )

    # ---- write a per-doc stats sidecar (handy for pilots) ----
    if args.stats_json:
        Path(args.stats_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.stats_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "source_project": args.source_project,
                    "output_project": args.output_project,
                    "contextual": args.contextual,
                    "default_year": args.default_year,
                    "llm_model": args.llm_model if args.contextual == "llm" else None,
                    "doc_preview_chars": args.doc_preview_chars,
                    "context_max_tokens": args.context_max_tokens,
                    "total_chunks_written": total_chunks_written,
                    "elapsed_s": chunks_total_elapsed,
                    "per_doc": per_doc_stats,
                },
                f,
                indent=2,
            )
        logger.info(f"Wrote indexing stats to {args.stats_json}")

    if args.skip_embeddings:
        logger.info("--skip-embeddings set; not building FAISS index.")
        return

    # ---- build FAISS index over new chunk texts ----
    logger.info("Building FAISS index over new (contextual) chunks ...")
    embedder = Embedder(EmbedderConfig())
    faiss_index = FaissIndex(
        index_file=tgt_faiss_path,
        dimension=DEFAULT_EMBEDDING_DIM,
        faiss_lock=None,
    )
    faiss_index.load_or_build_from_scratch()

    with tgt_db.session_factory() as session:
        new_chunks = session.query(Chunk).order_by(Chunk.id).all()
        ids = [c.id for c in new_chunks]
        texts = [c.chunk_text for c in new_chunks]

    t_emb = time.perf_counter()
    # batch embedding through the embedder's async API
    embeddings = await embedder.generate_embeddings_async(texts)
    embedding_ids = np.array(ids, dtype=np.int64)
    faiss_index.add_embeddings(embeddings, embedding_ids)
    logger.info(
        f"Indexed {len(ids)} embeddings into FAISS in "
        f"{time.perf_counter() - t_emb:.1f}s"
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Contextual Retrieval re-indexer (metadata or LLM-generated)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source-project", default=DEFAULT_SOURCE_PROJECT,
                        help=f"Source project name (default: {DEFAULT_SOURCE_PROJECT})")
    parser.add_argument("--output-project", required=True,
                        help="Output project name (creates {name}.rag_pipeline.db + .faiss_index.index)")
    parser.add_argument("--contextual", required=True, choices=["metadata", "llm"],
                        help="Contextual mode: 'metadata' = static prefix, 'llm' = per-chunk LLM context")
    parser.add_argument("--default-year", type=int, default=2023,
                        help="Default fiscal year for the metadata prefix (default: 2023)")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL,
                        help=f"LLM model for variant=llm (default: {DEFAULT_LLM_MODEL})")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help=f"Inference base URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--doc-preview-chars", type=int, default=DEFAULT_DOC_PREVIEW_CHARS,
                        help=f"Doc preview length in characters for variant=llm (default: {DEFAULT_DOC_PREVIEW_CHARS})")
    parser.add_argument("--context-max-tokens", type=int, default=DEFAULT_CONTEXT_MAX_TOKENS,
                        help=f"Max output tokens for context generation (default: {DEFAULT_CONTEXT_MAX_TOKENS})")
    parser.add_argument("--llm-concurrent", type=int, default=10,
                        help="Concurrent LLM calls per document for variant=llm (default: 10)")
    parser.add_argument("--document-limit", type=int, default=None,
                        help="Process only the first N documents (pilot mode)")
    parser.add_argument("--document-tickers", type=str, default=None,
                        help="Comma-separated list of tickers to process (pilot mode)")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Stop after writing chunks; skip FAISS build")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing target DB / FAISS")
    parser.add_argument("--resume", action="store_true",
                        help="Skip documents already present in the target DB (by file_name) and continue with the rest")
    parser.add_argument("--build-faiss-only", action="store_true",
                        help="Skip per-document processing entirely; just embed existing target-DB chunks into FAISS")
    parser.add_argument("--stats-json", type=str, default=None,
                        help="Optional path to write per-document timing stats")
    return parser.parse_args()


def main():
    args = parse_arguments()
    asyncio.run(reindex(args))


if __name__ == "__main__":
    main()
