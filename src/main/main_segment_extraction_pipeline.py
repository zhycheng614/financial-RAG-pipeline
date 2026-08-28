"""
Segment Extraction Pipeline using OpenAI Responses API.

This pipeline:
1. Takes a folder of document files (PDF or TXT) and a CSV file with queries
2. For PDF files: uploads once via Files API, references by file_id in
   Responses API (input_file) — the model sees the full document in context
3. For TXT files: uploads once via Files API into a vector store, then uses
   the file_search tool in Responses API — OpenAI handles chunking, embedding,
   and retrieval automatically (works for arbitrarily large files)
4. Outputs results to CSV files (one per document) in an "output" folder

NOTE: All files in the input folder must be the same type (all PDF or all TXT).
"""

import argparse
import csv
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from logging import DEBUG, INFO, basicConfig, getLogger
from pathlib import Path
from threading import Event, Lock
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from constants import OPENAI_API_KEY

logger = getLogger(__name__)

# Global shutdown event for graceful termination
_shutdown_event = Event()


# Configuration
DEFAULT_MODEL = "gpt-4.1"
DEFAULT_CONCURRENCY = 5
INDEX_COLUMN = "_original_index"  # Hidden column for sorting

# System prompt for extraction - kept short and direct
SYSTEM_PROMPT = """You are a precise data extraction assistant analyzing the attached financial document.

CRITICAL OUTPUT RULES:
1. Output ONLY the answer value directly - NEVER provide responses in full sentences
2. If the answer contains multiple values (a list), separate them with semicolons (;) NOT commas
3. Include units for numerical values if specified in the document

CORRECT OUTPUT EXAMPLES:
- "AAPL" (for stock ticker)
- "Consumer Electronics; Software Services; Cloud Computing" (for list)
- "$394.3 billion" (for revenue with unit)
- "3571" (for SIC code)

WRONG OUTPUT EXAMPLES (never do this):
- "The stock ticker is AAPL" ❌
- "Based on the document, the revenue is..." ❌
"""


def signal_handler(signum, frame):
    """Handle Ctrl+C signal for graceful shutdown."""
    logger.info("\n⚠ Received interrupt signal. Shutting down gracefully...")
    _shutdown_event.set()


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
try:
    # SIGTERM may not be available on Windows
    signal.signal(signal.SIGTERM, signal_handler)
except (AttributeError, OSError):
    pass  # SIGTERM not available on this platform


@dataclass
class QueryRow:
    """Represents a single query row from the CSV."""

    index: int  # Original index in the CSV (for ordering)
    row_data: Dict[str, str]  # Original row data
    question: str  # The question to ask
    question_column: str  # Name of the question column


@dataclass
class ProcessingResult:
    """Result of processing a single query."""

    query_row: QueryRow
    result: str
    error: Optional[str] = None


class DocumentFileClient:
    """
    Client for OpenAI Responses API supporting PDF and TXT documents.

    PDF mode (Files API + input_file):
    - Uploads the PDF once via OpenAI Files API (purpose="user_data")
    - References the file by file_id in each query via input_file
    - The model sees the full document in context

    TXT mode (Files API + vector store + file_search):
    - Uploads the TXT once via Files API (purpose="assistants")
    - Creates a vector store and indexes the file
    - Each query uses the file_search tool for retrieval
    - OpenAI handles chunking, embedding, and retrieval automatically
    - Works for arbitrarily large files (no context window limit)
    """

    # Regex to strip file_search citation markers like 【4:0†source】
    _CITATION_PATTERN = None

    def __init__(self, model: str = DEFAULT_MODEL):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model
        self.file_id: Optional[str] = None
        self.document_filename: Optional[str] = None
        self._vector_store_id: Optional[str] = None  # For TXT mode
        self._file_type: Optional[str] = None  # "pdf" or "txt"
        self._setup_complete = False

        # Lazy-compile the citation regex once
        if DocumentFileClient._CITATION_PATTERN is None:
            import re

            DocumentFileClient._CITATION_PATTERN = re.compile(
                r"【\d+:\d+†[^】]*】"
            )

    def setup_for_file(self, file_path: str, file_type: str) -> None:
        """
        Prepare the document for querying.

        For PDF: uploads to OpenAI via Files API (purpose="user_data").
        For TXT: uploads via Files API (purpose="assistants"), creates a
                 vector store, and indexes the file for file_search.

        Args:
            file_path: Path to the document file
            file_type: "pdf" or "txt"
        """
        # Check for shutdown
        if _shutdown_event.is_set():
            raise InterruptedError("Shutdown requested")

        self.document_filename = os.path.basename(file_path)
        self._file_type = file_type

        if file_type == "pdf":
            logger.info(f"Uploading PDF to OpenAI: {file_path}")
            with open(file_path, "rb") as f:
                file_response = self.client.files.create(
                    file=f, purpose="user_data"
                )
            self.file_id = file_response.id
            logger.info(f"PDF uploaded with file_id: {self.file_id}")
            logger.info(
                "Document ready for queries - file will be referenced by ID"
            )

        elif file_type == "txt":
            # Step 1: Upload file via Files API
            logger.info(f"Uploading TXT to OpenAI: {file_path}")
            with open(file_path, "rb") as f:
                file_response = self.client.files.create(
                    file=f, purpose="assistants"
                )
            self.file_id = file_response.id
            logger.info(f"TXT uploaded with file_id: {self.file_id}")

            # Step 2: Create a vector store
            store_name = f"pipeline-{self.document_filename}"
            logger.info(f"Creating vector store: {store_name}")
            vector_store = self.client.vector_stores.create(name=store_name)
            self._vector_store_id = vector_store.id
            logger.info(
                f"Vector store created: {self._vector_store_id}"
            )

            # Step 3: Add file to vector store and wait for indexing
            logger.info("Indexing file in vector store (this may take a moment)...")
            vs_file = self.client.vector_stores.files.create_and_poll(
                vector_store_id=self._vector_store_id,
                file_id=self.file_id,
            )
            if vs_file.status != "completed":
                raise RuntimeError(
                    f"Vector store file indexing failed: "
                    f"status={vs_file.status}, "
                    f"last_error={getattr(vs_file, 'last_error', 'unknown')}"
                )
            logger.info(
                "File indexed successfully. "
                "Queries will use file_search for retrieval."
            )

        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        self._setup_complete = True

    def query(self, question: str) -> str:
        """
        Execute a query against the loaded document.

        For PDF: references file_id via input_file in the Responses API.
        For TXT: uses the file_search tool with the vector store so OpenAI
                 retrieves relevant chunks automatically.

        Args:
            question: The question to ask about the document

        Returns:
            The extracted answer
        """
        # Check for shutdown
        if _shutdown_event.is_set():
            raise InterruptedError("Shutdown requested")

        if not self._setup_complete:
            raise RuntimeError(
                "Document not loaded. Call setup_for_file() first."
            )

        # Check for shutdown before API call
        if _shutdown_event.is_set():
            raise InterruptedError("Shutdown requested")

        # Make the API call using Responses API
        try:
            if self._file_type == "pdf":
                # PDF mode: reference file directly in context
                if not self.file_id:
                    raise RuntimeError("PDF file_id is missing.")
                response = self.client.responses.create(
                    model=self.model,
                    instructions=SYSTEM_PROMPT,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_file",
                                    "file_id": self.file_id,
                                },
                                {
                                    "type": "input_text",
                                    "text": f"Question: {question}",
                                },
                            ],
                        }
                    ],
                )
            else:
                # TXT mode: use file_search tool with vector store
                response = self.client.responses.create(
                    model=self.model,
                    instructions=SYSTEM_PROMPT,
                    tools=[
                        {
                            "type": "file_search",
                            "vector_store_ids": [self._vector_store_id],
                        }
                    ],
                    input=[
                        {
                            "role": "user",
                            "content": f"Question: {question}",
                        }
                    ],
                )

            # Extract the response
            if response.output_text:
                result = response.output_text.strip()
                # Strip citation markers (e.g. 【4:0†source】) from file_search
                if self._file_type == "txt" and self._CITATION_PATTERN:
                    result = self._CITATION_PATTERN.sub("", result).strip()
                return result

            return " "  # Return single space if no result

        except Exception as e:
            logger.error(f"API call failed: {e}")
            raise

    def cleanup(self) -> None:
        """
        Clean up resources on OpenAI.

        PDF mode: deletes the uploaded file.
        TXT mode: deletes the vector store first, then the uploaded file.
        """
        # Delete vector store (TXT mode)
        if self._vector_store_id:
            logger.info(
                f"Deleting vector store: {self._vector_store_id}"
            )
            try:
                self.client.vector_stores.delete(
                    vector_store_id=self._vector_store_id
                )
                logger.info(
                    f"Vector store deleted: {self._vector_store_id}"
                )
            except Exception as e:
                logger.warning(f"Failed to delete vector store: {e}")

        # Delete uploaded file (both modes)
        if self.file_id:
            logger.info(f"Deleting uploaded file: {self.file_id}")
            try:
                self.client.files.delete(self.file_id)
                logger.info(f"File deleted: {self.file_id}")
            except Exception as e:
                logger.warning(f"Failed to delete file: {e}")

        self.file_id = None
        self._vector_store_id = None
        self.document_filename = None
        self._file_type = None
        self._setup_complete = False


class StreamingCSVWriter:
    """
    Thread-safe CSV writer that writes results row by row.
    Uses a lock to prevent race conditions during concurrent writes.
    Includes an index column for later sorting.
    Sanitizes text to prevent CSV structure issues.

    Uses open-write-close approach for each row to allow other programs
    to read the file while processing is ongoing (important for Windows).
    """

    def __init__(
        self,
        output_path: str,
        headers: List[str],
        result_column: str = "result",
    ):
        self.output_path = output_path
        # Add index column at the beginning for sorting
        self.internal_headers = [INDEX_COLUMN] + headers + [result_column]
        self.output_headers = headers + [
            result_column
        ]  # Headers for final output
        self.result_column = result_column
        self._lock = Lock()
        self._rows_written = 0
        self._initialized = False

    @staticmethod
    def _sanitize_text(value: Any) -> str:
        """
        Sanitize text for CSV output:
        1. Escape newlines so CSV stays one row per record
        2. Convert Unicode special characters to ASCII equivalents

        Args:
            value: The value to sanitize (can be any type)

        Returns:
            Sanitized string safe for CSV output
        """
        if value is None:
            return ""
        str_value = str(value)

        # Replace actual newlines with literal \n
        str_value = (
            str_value.replace("\r\n", "\\n")
            .replace("\n", "\\n")
            .replace("\r", "\\n")
        )

        # Convert Unicode special characters to ASCII equivalents
        unicode_replacements = {
            # Quotation marks
            "\u2018": "'",  # ' left single quotation mark
            "\u2019": "'",  # ' right single quotation mark (curly apostrophe)
            "\u201c": '"',  # " left double quotation mark
            "\u201d": '"',  # " right double quotation mark
            # Dashes
            "\u2013": "-",  # – en dash
            "\u2014": "--",  # — em dash
            # Math symbols
            "\u00d7": "x",  # × multiplication sign
            "\u00f7": "/",  # ÷ division sign
            "\u2212": "-",  # − minus sign
            "\u00b1": "+/-",  # ± plus-minus sign
            # Other common symbols
            "\u2026": "...",  # … ellipsis
            "\u2022": "-",  # • bullet
            "\u00b0": " deg",  # ° degree sign
            "\u00a0": " ",  # non-breaking space
            # Additional financial/document symbols
            "\u201a": ",",  # ‚ single low quotation mark
            "\u201e": '"',  # „ double low quotation mark
            "\u2032": "'",  # ′ prime
            "\u2033": '"',  # ″ double prime
            "\u2039": "<",  # ‹ single left angle quote
            "\u203a": ">",  # › single right angle quote
            "\u00ab": "<<",  # « double left angle quote
            "\u00bb": ">>",  # » double right angle quote
            "\u2010": "-",  # ‐ hyphen
            "\u2011": "-",  # ‑ non-breaking hyphen
            "\u2015": "--",  # ― horizontal bar
            "\u00ad": "",  # soft hyphen (remove)
        }

        for unicode_char, ascii_equiv in unicode_replacements.items():
            str_value = str_value.replace(unicode_char, ascii_equiv)

        return str_value

    def open(self) -> "StreamingCSVWriter":
        """Initialize the CSV file by writing headers."""
        with self._lock:
            if self._initialized:
                return self

            # Write headers to new file with UTF-8 BOM for Windows compatibility
            with open(
                self.output_path, "w", newline="", encoding="utf-8-sig"
            ) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=self.internal_headers,
                    quoting=csv.QUOTE_ALL,
                )
                writer.writeheader()

            self._initialized = True
            logger.info(f"Created output CSV: {self.output_path}")
        return self

    def close(self) -> None:
        """Mark the writer as closed (file is already closed after each write)."""
        with self._lock:
            self._initialized = False

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def write_row(self, query_row: QueryRow, result: str) -> None:
        """
        Write a single result row to the CSV.
        Thread-safe - uses lock to prevent concurrent write issues.
        Sanitizes the result to prevent CSV structure issues.

        Opens file in append mode, writes row, then closes - allowing
        other programs to read the file between writes.
        """
        with self._lock:
            if not self._initialized:
                logger.warning("Attempted to write to uninitialized CSV file")
                return

            # Build row data with sanitized values
            row_data = {}
            for key, value in query_row.row_data.items():
                row_data[key] = self._sanitize_text(value)

            # Sanitize the result to handle newlines and special characters
            row_data[self.result_column] = self._sanitize_text(result)
            row_data[INDEX_COLUMN] = query_row.index  # Add index for sorting

            # Open in append mode, write, and close immediately
            # This allows other programs to read the file between writes
            try:
                with open(
                    self.output_path, "a", newline="", encoding="utf-8-sig"
                ) as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=self.internal_headers,
                        quoting=csv.QUOTE_ALL,
                    )
                    writer.writerow(row_data)
                self._rows_written += 1
                logger.debug(
                    f"Written row {self._rows_written} to CSV (index={query_row.index})"
                )
            except Exception as e:
                logger.error(f"Failed to write row to CSV: {e}")
                import traceback

                traceback.print_exc()

    @property
    def rows_written(self) -> int:
        """Get the number of rows written so far."""
        with self._lock:
            return self._rows_written


def sort_csv_by_original_order(csv_path: str) -> None:
    """
    Sort a CSV file by the original index column and remove the index column.
    This restores the original query order after concurrent processing.
    """
    logger.info(f"Sorting CSV by original order: {csv_path}")

    # Read all rows
    rows = []
    headers = []

    try:
        with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            for row in reader:
                rows.append(row)
    except Exception as e:
        logger.error(f"Failed to read CSV for sorting: {e}")
        return

    if not rows:
        logger.warning("No rows to sort")
        return

    # Sort by original index
    try:
        rows.sort(key=lambda r: int(r.get(INDEX_COLUMN, 0)))
    except (ValueError, TypeError) as e:
        logger.error(f"Failed to sort rows: {e}")
        return

    # Remove the index column from headers and rows
    output_headers = [h for h in headers if h != INDEX_COLUMN]

    # Rewrite the file without the index column (with UTF-8 BOM for Windows)
    try:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=output_headers,
                quoting=csv.QUOTE_ALL,  # Quote all fields to handle special chars
            )
            writer.writeheader()
            for row in rows:
                # Remove index column from row
                output_row = {k: v for k, v in row.items() if k != INDEX_COLUMN}
                writer.writerow(output_row)

        logger.info(f"CSV sorted and index column removed: {csv_path}")
    except Exception as e:
        logger.error(f"Failed to write sorted CSV: {e}")


def load_queries_from_csv(
    csv_path: str,
) -> Tuple[List[QueryRow], List[str], str]:
    """
    Load queries from a CSV file.

    Returns:
        Tuple of (list of QueryRow objects, list of column headers, question column name)
    """
    queries = []
    headers = []
    question_column = None

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

        # Find the question column (look for 'question', 'query', or similar)
        question_candidates = [
            "question",
            "query",
            "questions",
            "queries",
            "prompt",
        ]
        for candidate in question_candidates:
            if candidate in headers:
                question_column = candidate
                break
            # Case-insensitive check
            for h in headers:
                if h.lower() == candidate:
                    question_column = h
                    break
            if question_column:
                break

        if not question_column:
            raise ValueError(
                f"Could not find question column. Available columns: {headers}"
            )

        for idx, row in enumerate(reader):
            question = row.get(question_column, "").strip()
            if question:
                queries.append(
                    QueryRow(
                        index=idx,
                        row_data=dict(row),
                        question=question,
                        question_column=question_column,
                    )
                )

    logger.info(f"Loaded {len(queries)} queries from {csv_path}")
    return queries, headers, question_column


def get_document_files(folder_path: str) -> Tuple[List[str], str]:
    """
    Get all supported document files from a folder.

    All files must be the same type (all PDF or all TXT).

    Returns:
        Tuple of (list of file paths, file type string "pdf" or "txt")

    Raises:
        ValueError: If folder contains mixed file types or no supported files
    """
    pdf_files = []
    txt_files = []

    for file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file)
        if file.lower().endswith(".pdf"):
            pdf_files.append(full_path)
        elif file.lower().endswith(".txt"):
            txt_files.append(full_path)

    if pdf_files and txt_files:
        raise ValueError(
            f"Input folder contains mixed file types "
            f"({len(pdf_files)} PDF, {len(txt_files)} TXT). "
            f"All files must be the same type (all PDF or all TXT)."
        )

    if pdf_files:
        pdf_files.sort()
        logger.info(f"Found {len(pdf_files)} PDF files in {folder_path}")
        return pdf_files, "pdf"

    if txt_files:
        txt_files.sort()
        logger.info(f"Found {len(txt_files)} TXT files in {folder_path}")
        return txt_files, "txt"

    raise ValueError(
        f"No document files (PDF or TXT) found in: {folder_path}"
    )


def process_single_query(
    client: DocumentFileClient,
    query_row: QueryRow,
    writer: StreamingCSVWriter,
    total_queries: int,
) -> ProcessingResult:
    """
    Process a single query and write result to CSV.
    This function is designed to be called concurrently.

    Args:
        client: The document client (shared, thread-safe)
        query_row: The query to process
        writer: Thread-safe CSV writer
        total_queries: Total number of queries (for logging)

    Returns:
        ProcessingResult with the query result
    """
    query_idx = query_row.index + 1  # 1-based for display

    # Check for shutdown - skip this query but don't write anything
    # (query hasn't started yet, so no partial result)
    if _shutdown_event.is_set():
        logger.debug(f"[{query_idx}] Skipped due to shutdown")
        return ProcessingResult(
            query_row=query_row, result=" ", error="Shutdown - skipped"
        )

    try:
        result = client.query(query_row.question)

        # Check for shutdown AFTER API call - don't write if shutting down
        if _shutdown_event.is_set():
            logger.debug(
                f"[{query_idx}] Shutdown detected after API call, skipping write"
            )
            return ProcessingResult(
                query_row=query_row,
                result=" ",
                error="Shutdown - result discarded",
            )

        # Ensure result is valid (single space if empty)
        if not result or not result.strip():
            result = " "

        logger.info(
            f"[{query_idx}/{total_queries}] ✓ {query_row.question[:40]}... → {result[:50]}{'...' if len(result) > 50 else ''}"
        )

        # Write result immediately (thread-safe)
        writer.write_row(query_row, result)

        return ProcessingResult(query_row=query_row, result=result)

    except InterruptedError:
        # Shutdown requested during query - write placeholder
        logger.info(f"[{query_idx}/{total_queries}] ⚠ Interrupted")
        writer.write_row(query_row, " ")
        return ProcessingResult(
            query_row=query_row, result=" ", error="Shutdown - interrupted"
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(
            f"[{query_idx}/{total_queries}] ✗ {query_row.question[:40]}... → Error: {error_msg}"
        )

        # Write single space for errors
        writer.write_row(query_row, " ")

        return ProcessingResult(
            query_row=query_row, result=" ", error=error_msg
        )


def process_document_with_queries(
    document_path: str,
    queries: List[QueryRow],
    output_path: str,
    headers: List[str],
    file_type: str,
    model: str = DEFAULT_MODEL,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> bool:
    """
    Process a single document file with all queries and write results to CSV.
    Uses concurrent processing for queries.

    Args:
        document_path: Path to the document file (PDF or TXT)
        queries: List of QueryRow objects
        output_path: Path for the output CSV file
        headers: Column headers for the CSV
        file_type: "pdf" or "txt"
        model: OpenAI model to use
        concurrency: Number of concurrent query threads

    Returns:
        True if completed successfully, False if interrupted
    """
    document_name = os.path.basename(document_path)
    mode_desc = (
        "Files API + input_file (full document in context)"
        if file_type == "pdf"
        else "Files API + vector store + file_search (retrieval)"
    )
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing document: {document_name}")
    logger.info(f"Queries: {len(queries)}, Concurrency: {concurrency}")
    logger.info(f"Mode: {mode_desc}")
    logger.info(f"{'='*60}")

    # Step 1: Create output CSV file FIRST (before any OpenAI operations)
    writer = StreamingCSVWriter(output_path, headers)
    writer.open()
    logger.info(f"Output file created: {output_path}")

    client = DocumentFileClient(model=model)
    executor = None
    interrupted = False

    try:
        # Check for shutdown before starting
        if _shutdown_event.is_set():
            logger.info("Shutdown requested before processing started")
            return False

        # Step 2: Prepare document (upload for PDF, read for TXT)
        client.setup_for_file(document_path, file_type)

        # Step 3: Process queries concurrently
        # Each query references the uploaded file by ID
        total_queries = len(queries)
        completed = 0
        errors = 0

        executor = ThreadPoolExecutor(max_workers=concurrency)

        # Submit all queries
        future_to_query = {
            executor.submit(
                process_single_query,
                client,
                query_row,
                writer,
                total_queries,
            ): query_row
            for query_row in queries
        }

        # Process results as they complete
        for future in as_completed(future_to_query):
            # Check for shutdown
            if _shutdown_event.is_set():
                logger.info("Shutdown requested, cancelling remaining tasks...")
                interrupted = True
                break

            query_row = future_to_query[future]
            try:
                result = future.result(
                    timeout=1
                )  # Short timeout to check shutdown
                completed += 1
                if result.error:
                    errors += 1
            except Exception as e:
                if not _shutdown_event.is_set():
                    logger.error(f"Unexpected error processing query: {e}")
                errors += 1
                completed += 1

        if interrupted:
            logger.info(f"⚠ Processing interrupted for {document_name}")
            logger.info(f"  Completed: {completed}/{total_queries}")
        else:
            logger.info(f"\n✓ Completed processing {document_name}")
            logger.info(f"  Total: {completed}, Errors: {errors}")

        logger.info(f"  Results written to: {output_path}")
        return not interrupted

    except InterruptedError:
        logger.info(f"⚠ Processing interrupted for {document_name}")
        return False
    except Exception as e:
        logger.error(f"Error processing {document_name}: {e}")
        raise
    finally:
        # Always cleanup resources - order matters!

        # 1. Shutdown executor immediately - don't wait for in-flight tasks
        if executor is not None:
            # cancel_futures=True cancels pending tasks
            # wait=False means don't block waiting for running tasks
            executor.shutdown(wait=False, cancel_futures=True)
            logger.info(
                "ThreadPoolExecutor shut down (not waiting for in-flight requests)"
            )

        # 2. Close CSV writer
        writer.close()
        logger.debug("CSV writer closed")

        # 3. Sort the CSV by original order (only if we have results)
        rows_written = writer.rows_written
        logger.info(f"Rows written before sorting: {rows_written}")
        if rows_written > 0:
            sort_csv_by_original_order(output_path)
        else:
            logger.warning("No rows were written to CSV")

        # 4. Cleanup client (delete uploaded file from OpenAI)
        client.cleanup()


def run_pipeline(
    document_folder: str,
    queries_csv: str,
    model: str = DEFAULT_MODEL,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> None:
    """
    Run the full segment extraction pipeline.

    Args:
        document_folder: Path to folder containing document files (PDF or TXT)
        queries_csv: Path to CSV file with queries
        model: OpenAI model to use
        concurrency: Number of concurrent query threads per document
    """
    logger.info("Starting Segment Extraction Pipeline")
    logger.info(f"Document Folder: {document_folder}")
    logger.info(f"Queries CSV: {queries_csv}")
    logger.info(f"Model: {model}")
    logger.info(f"Concurrency: {concurrency}")

    # Validate inputs
    if not os.path.isdir(document_folder):
        raise ValueError(f"Document folder does not exist: {document_folder}")
    if not os.path.isfile(queries_csv):
        raise ValueError(f"Queries CSV does not exist: {queries_csv}")

    # Load queries
    queries, headers, question_column = load_queries_from_csv(queries_csv)
    if not queries:
        raise ValueError("No queries found in CSV file")

    # Get document files (all must be same type)
    document_files, file_type = get_document_files(document_folder)
    mode_desc = (
        "Files API + input_file (full document in context)"
        if file_type == "pdf"
        else "Files API + vector store + file_search (retrieval)"
    )
    logger.info(f"File type: {file_type.upper()}, Mode: {mode_desc}")

    # Create output directory in the same folder as the queries CSV
    queries_dir = os.path.dirname(os.path.abspath(queries_csv))
    output_dir = os.path.join(queries_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Process each document
    total_documents = len(document_files)
    for doc_idx, document_path in enumerate(document_files, 1):
        # Check for shutdown before each document
        if _shutdown_event.is_set():
            logger.info("Shutdown requested, stopping pipeline")
            break

        document_name = Path(document_path).stem
        output_filename = f"{document_name}_results.csv"
        output_path = os.path.join(output_dir, output_filename)

        logger.info(f"\n[Document {doc_idx}/{total_documents}] {document_name}")

        try:
            success = process_document_with_queries(
                document_path=document_path,
                queries=queries,
                output_path=output_path,
                headers=headers,
                file_type=file_type,
                model=model,
                concurrency=concurrency,
            )
            if not success:
                logger.info("Stopping pipeline due to interruption")
                break
        except Exception as e:
            logger.error(f"Failed to process {document_name}: {str(e)}")
            import traceback

            traceback.print_exc()
            # Continue with next document unless shutdown requested
            if _shutdown_event.is_set():
                break
            continue

    logger.info("\n" + "=" * 60)
    if _shutdown_event.is_set():
        logger.info("Pipeline stopped (interrupted by user)")
    else:
        logger.info("Pipeline completed!")
    logger.info(f"Results saved in: {output_dir}")
    logger.info("=" * 60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Segment Extraction Pipeline - Extract data from documents (PDF/TXT) using OpenAI Responses API"
    )

    parser.add_argument(
        "--document-folder",
        type=str,
        required=True,
        help="Path to folder containing document files (all PDF or all TXT, no mixing)",
    )

    parser.add_argument(
        "--queries-csv",
        type=str,
        required=True,
        help="Path to CSV file containing queries (must have a 'question' column)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_MODEL})",
    )

    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Number of concurrent query threads per document (default: {DEFAULT_CONCURRENCY})",
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = DEBUG if args.verbose else INFO
    basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        run_pipeline(
            document_folder=args.document_folder,
            queries_csv=args.queries_csv,
            model=args.model,
            concurrency=args.concurrency,
        )
    except KeyboardInterrupt:
        # This should not be reached due to signal handler, but just in case
        logger.info("\nPipeline interrupted by user")
        sys.exit(130)  # Standard exit code for Ctrl+C
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise
    finally:
        # Ensure clean exit
        if _shutdown_event.is_set():
            sys.exit(130)


if __name__ == "__main__":
    main()
