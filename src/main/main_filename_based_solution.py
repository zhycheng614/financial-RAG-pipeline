#!/usr/bin/env python
"""
Filename-based query processing pipeline for financial reports.

This script:
1. Takes queries from multiple sources (CLI, CSV, or HuggingFace dataset)
2. Extracts ticker symbols and years from queries using LLM
3. Locates corresponding document files (PDF or TXT) directly by filename
4. Sends queries with the document files to the LLM for answer generation
5. Outputs results to terminal or CSV file

Unlike the RAG pipeline, this approach does NOT use vector search or keyword search.
Instead, it relies on a naming convention: {data_dir}/{year}/{ticker_symbol}.pdf (or .txt)
"""

import argparse
import asyncio
import base64
import csv
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from beans.inference_facade import InferenceFacade
from data_classes.inference_parameters import InferenceParameters
from data_classes.ticker_extraction_output import TickerExtractionOutput
from main.query_io import (
    CLIInputSource,
    CSVInputSource,
    HuggingFaceInputSource,
    QueryInputSource,
    QueryItem,
    QueryResult,
    create_input_source,
    output_to_csv,
    output_to_terminal,
)
from prompts import (
    FILE_BASED_ANSWER_SYSTEM_PROMPT,
    FILE_BASED_ANSWER_USER_PROMPT,
)
from service.rewriting_service import RewritingService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default inference settings
DEFAULT_MODEL = "gpt-4.1"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_YEAR = datetime.now().year


class StreamingCSVWriter:
    """
    A thread-safe CSV writer that writes results incrementally as they complete.
    Uses an asyncio lock to prevent race conditions when multiple coroutines
    write concurrently.
    """

    def __init__(self, filepath: str, fieldnames: List[str]):
        """
        Initialize the streaming CSV writer.

        Args:
            filepath: Path to the output CSV file
            fieldnames: List of column names for the CSV
        """
        self.filepath = filepath
        self.fieldnames = fieldnames
        self.lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self):
        """Initialize the CSV file with headers."""
        async with self.lock:
            if not self._initialized:
                # Ensure parent directory exists
                Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)

                # Write headers with QUOTE_ALL to properly handle multiline content
                with open(
                    self.filepath, "w", newline="", encoding="utf-8"
                ) as f:
                    writer = csv.DictWriter(
                        f, fieldnames=self.fieldnames, quoting=csv.QUOTE_ALL
                    )
                    writer.writeheader()

                self._initialized = True
                logger.info(f"Initialized CSV output file: {self.filepath}")

    @staticmethod
    def _sanitize_text(value) -> str:
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
        }

        for unicode_char, ascii_equiv in unicode_replacements.items():
            str_value = str_value.replace(unicode_char, ascii_equiv)

        return str_value

    async def write_result(self, result: "QueryResult", original_row: int):
        """
        Write a single result to the CSV file.

        Args:
            result: The QueryResult to write
            original_row: The original row number (1-based) in the input source
        """
        async with self.lock:
            if not self._initialized:
                raise RuntimeError(
                    "StreamingCSVWriter not initialized. Call initialize() first."
                )

            # Build the row data with sanitized text (escaped newlines + ASCII conversion)
            row = {
                "original_row": original_row,
                "query_id": self._sanitize_text(result.query_item.query_id),
                "query": self._sanitize_text(result.query_item.query),
                "tickers": self._sanitize_text(
                    result.metadata.get("tickers", "")
                ),
                "file_paths": self._sanitize_text(
                    result.metadata.get("file_paths", "")
                ),
                "missing_files": self._sanitize_text(
                    result.metadata.get("missing_files", "")
                ),
                "final_answer": self._sanitize_text(result.final_answer),
                "error": self._sanitize_text(result.error),
            }

            # Filter to only include fields in fieldnames
            row = {k: v for k, v in row.items() if k in self.fieldnames}

            # Append to CSV with QUOTE_ALL to properly handle special characters
            with open(self.filepath, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=self.fieldnames, quoting=csv.QUOTE_ALL
                )
                writer.writerow(row)

            logger.debug(f"Written result for row {original_row} to CSV")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Filename-based Financial Report Query Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single query from CLI
  python main/main_filename_based_solution.py --query "What was AAL's revenue in 2023?" --data-dir ./reports

  # Query with custom default year
  python main/main_filename_based_solution.py --query "What is Apple's strategy?" --data-dir ./reports --default-year 2024

  # Queries from CSV file
  python main/main_filename_based_solution.py --csv queries.csv --column question --data-dir ./reports

  # Queries from CSV with ID column
  python main/main_filename_based_solution.py --csv queries.csv --column question --id-column query_id --data-dir ./reports

  # Queries from HuggingFace dataset
  python main/main_filename_based_solution.py --hf-dataset my_dataset --column question --split validation --data-dir ./reports

  # Output to CSV
  python main/main_filename_based_solution.py --csv queries.csv --column question --data-dir ./reports --output-csv results.csv

  # Adjust concurrency
  python main/main_filename_based_solution.py --csv queries.csv --column question --data-dir ./reports --concurrent 5

  # Process specific row range (rows 101-200)
  python main/main_filename_based_solution.py --csv queries.csv --column question --data-dir ./reports --start-row 101 --end-row 200

  # Auto-generate output filename (will create ./output/{csv_name}-101-to-200-{model}-{timestamp}.csv)
  python main/main_filename_based_solution.py --csv queries.csv --column question --data-dir ./reports --start-row 101 --end-row 200 --output-csv

Input Options:
  • CLI: Single query provided via --query argument
  • CSV: Multiple queries from a CSV file column
  • HuggingFace: Multiple queries from a HuggingFace dataset column

Output Options:
  • Terminal: Default, prints results to console
  • CSV: Save results to CSV file with --output-csv

File Naming Convention:
  The pipeline expects document files (PDF or TXT) to be organized as:
  {data-dir}/{year}/{ticker_symbol}.pdf (or .txt)
  
  Example:
  ./reports/2023/AAPL.pdf
  ./reports/2023/AAL.txt
  ./reports/2024/MSFT.pdf
        """,
    )

    # Input source arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--query", type=str, help="Single query from command line"
    )
    input_group.add_argument(
        "--csv",
        type=str,
        metavar="PATH",
        help="Path to CSV file containing queries",
    )
    input_group.add_argument(
        "--hf-dataset",
        type=str,
        metavar="DATASET_ID",
        help="HuggingFace dataset ID",
    )

    # Data directory (required)
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to directory containing financial report documents (PDF or TXT) organized as {year}/{ticker}.{ext}",
    )

    # Default year for queries without explicit year
    parser.add_argument(
        "--default-year",
        type=int,
        default=DEFAULT_YEAR,
        metavar="YEAR",
        help=f"Default year to use when query does not specify a year (default: {DEFAULT_YEAR})",
    )

    # Column and split arguments
    parser.add_argument(
        "--column",
        type=str,
        metavar="NAME",
        help="Column name containing queries (required for --csv and --hf-dataset)",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        metavar="NAME",
        help="Column name for query IDs (optional, defaults to row index)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        metavar="NAME",
        help="Dataset split for HuggingFace datasets (default: train)",
    )

    # Row range arguments (for CSV and HuggingFace datasets)
    parser.add_argument(
        "--start-row",
        type=int,
        default=None,
        metavar="N",
        help="Starting row number (1-based, inclusive). Only applies to --csv and --hf-dataset",
    )
    parser.add_argument(
        "--end-row",
        type=int,
        default=None,
        metavar="N",
        help="Ending row number (1-based, inclusive). Only applies to --csv and --hf-dataset",
    )

    # Output arguments
    parser.add_argument(
        "--output-csv",
        type=str,
        nargs="?",
        const="",  # Value when flag is provided without argument
        default=None,  # Value when flag is not provided
        metavar="PATH",
        help="Output results to CSV file. If flag is provided without path, auto-generates filename in ./output/",
    )

    # Processing configuration
    parser.add_argument(
        "--concurrent",
        type=int,
        default=3,
        metavar="N",
        help="Number of concurrent query processing tasks (default: 3)",
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        metavar="NAME",
        help=f"Model name for answer generation (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        metavar="URL",
        help=f"Base URL for inference service (default: {DEFAULT_BASE_URL})",
    )

    # Logging
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )

    return parser.parse_args()


def encode_file_to_base64(file_path: str) -> str:
    """
    Encode a file to base64 string.

    Args:
        file_path: Path to the file (PDF or TXT)

    Returns:
        Base64 encoded string of the file content
    """
    with open(file_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def get_mime_type(file_path: str) -> str:
    """
    Get the MIME type for a file based on its extension.

    Args:
        file_path: Path to the file

    Returns:
        MIME type string
    """
    ext = os.path.splitext(file_path)[1].lower()
    mime_types = {
        ".pdf": "application/pdf",
        ".txt": "text/plain",
    }
    return mime_types.get(ext, "application/octet-stream")


# Supported file extensions (in order of preference)
SUPPORTED_EXTENSIONS = [".pdf", ".txt"]


def find_document_file(base_dir: str, year: int, ticker: str) -> Optional[str]:
    """
    Find a document file for a given ticker and year, trying supported extensions.

    Args:
        base_dir: Base data directory
        year: Year to look in
        ticker: Ticker symbol

    Returns:
        Path to the found file, or None if no file exists
    """
    for ext in SUPPORTED_EXTENSIONS:
        path = os.path.join(base_dir, str(year), f"{ticker}{ext}")
        if os.path.exists(path) and os.path.isfile(path):
            return path
    return None


def validate_file_paths(file_paths: List[str]) -> Tuple[List[str], List[str]]:
    """
    Validate that all file paths exist.

    Args:
        file_paths: List of file paths to validate

    Returns:
        Tuple of (valid_paths, missing_paths)
    """
    valid_paths = []
    missing_paths = []

    for path in file_paths:
        if os.path.exists(path) and os.path.isfile(path):
            valid_paths.append(path)
        else:
            missing_paths.append(path)

    return valid_paths, missing_paths


def sanitize_messages_for_logging(messages: List[dict]) -> List[dict]:
    """
    Sanitize messages for logging by removing large base64 file data.

    Args:
        messages: List of message dicts that may contain file data

    Returns:
        Sanitized messages safe for logging (base64 data replaced with placeholder)
    """
    sanitized = []
    for msg in messages:
        sanitized_msg = {"role": msg.get("role", "unknown")}
        content = msg.get("content")

        if isinstance(content, str):
            # Simple text content
            sanitized_msg["content"] = (
                content[:200] + "..." if len(content) > 200 else content
            )
        elif isinstance(content, list):
            # Multi-part content (may contain files)
            sanitized_content = []
            for part in content:
                if part.get("type") == "file":
                    # Replace file data with placeholder
                    sanitized_content.append(
                        {
                            "type": "file",
                            "file": {
                                "filename": part.get("file", {}).get(
                                    "filename", "unknown"
                                ),
                                "file_data": "[BASE64_DATA_REDACTED]",
                            },
                        }
                    )
                elif part.get("type") == "text":
                    text = part.get("text", "")
                    sanitized_content.append(
                        {
                            "type": "text",
                            "text": (
                                text[:200] + "..." if len(text) > 200 else text
                            ),
                        }
                    )
                else:
                    sanitized_content.append(part)
            sanitized_msg["content"] = sanitized_content
        else:
            sanitized_msg["content"] = str(content)[:200] if content else None

        sanitized.append(sanitized_msg)
    return sanitized


async def process_single_query(
    query_item: QueryItem,
    data_dir: str,
    default_year: int,
    rewriting_service: RewritingService,
    inference_facade: InferenceFacade,
    progress_info: Optional[Tuple[int, int, int]] = None,
) -> QueryResult:
    """
    Process a single query using the filename-based approach.

    Steps:
    1. Extract ticker symbols and years from the query
    2. Find document files (PDF or TXT) based on ticker/year
    3. Validate file paths exist
    4. Send query with document files to LLM for answer generation

    Args:
        query_item: Query to process
        data_dir: Base directory containing financial report PDFs
        default_year: Default year to use if not specified in query
        rewriting_service: Service for extracting ticker info
        inference_facade: Service for LLM inference
        progress_info: Optional tuple of (current_row, start_row, end_row) for progress logging

    Returns:
        QueryResult with the final answer
    """
    result = QueryResult(query_item=query_item)

    try:
        # Build progress string if progress_info is provided
        progress_str = ""
        if progress_info:
            current_row, start_row, end_row = progress_info
            progress_num = current_row - start_row + 1
            total_num = end_row - start_row + 1
            progress_str = f" [Progress: {progress_num}/{total_num}]"

        logger.info(
            f"Processing query [{query_item.query_id}]{progress_str}: {query_item.query[:100]}..."
        )

        # Step 1: Extract ticker symbols and years from query
        ticker_extraction = await rewriting_service.extract_ticker_info_async(
            query_item.query, default_year
        )

        if ticker_extraction.is_empty():
            logger.warning(
                f"No ticker symbols extracted for query [{query_item.query_id}]"
            )
            result.final_answer = (
                "Could not identify any company ticker symbols from the query."
            )
            result.error = "No ticker symbols extracted"
            return result

        # Log extracted info
        for item in ticker_extraction.items:
            logger.info(
                f"Extracted: {item.ticker_symbol} for years {item.years}"
            )

        # Store tickers in metadata
        result.metadata["tickers"] = ",".join(
            [item.ticker_symbol for item in ticker_extraction.items]
        )

        # Step 2: Find document files (PDF or TXT) with fallback logic
        # For each ticker, try to find files. If none found, fallback to default year.
        valid_paths = []
        missing_tickers_years = (
            []
        )  # Track ticker/year combos that were not found
        failed_tickers = (
            []
        )  # Tickers for which no files could be found even with fallback

        for ticker_info in ticker_extraction.items:
            ticker = ticker_info.ticker_symbol
            years = ticker_info.years

            # Try to find files for this ticker across requested years
            ticker_valid = []
            ticker_missing_years = []

            for year in years:
                found_path = find_document_file(data_dir, year, ticker)
                if found_path:
                    ticker_valid.append(found_path)
                else:
                    ticker_missing_years.append(year)

            if ticker_valid:
                # At least some files found for this ticker
                valid_paths.extend(ticker_valid)
                if ticker_missing_years:
                    missing_tickers_years.extend(
                        [(ticker, y) for y in ticker_missing_years]
                    )
                    logger.warning(
                        f"Some files missing for ticker {ticker} in years: {ticker_missing_years}. "
                        f"Found: {ticker_valid}"
                    )
            else:
                # No files found for any of the extracted years - try fallback to default year
                logger.warning(
                    f"No files found for ticker {ticker} with years {years}. "
                    f"Attempting fallback to default year {default_year}..."
                )

                fallback_path = find_document_file(
                    data_dir, default_year, ticker
                )

                if fallback_path:
                    logger.info(f"Fallback successful: found {fallback_path}")
                    valid_paths.append(fallback_path)
                    missing_tickers_years.extend([(ticker, y) for y in years])
                    result.metadata["fallback_used"] = "true"
                else:
                    # Fallback also failed - this ticker has no available files
                    logger.error(
                        f"Fallback failed for ticker {ticker} in year {default_year}. "
                        f"No files available for this company."
                    )
                    failed_tickers.append(ticker)
                    missing_tickers_years.extend([(ticker, y) for y in years])
                    missing_tickers_years.append((ticker, default_year))

        # Store file paths in metadata
        result.metadata["file_paths"] = (
            ",".join(valid_paths) if valid_paths else ""
        )
        if missing_tickers_years:
            result.metadata["missing_files"] = ",".join(
                [f"{t}/{y}" for t, y in missing_tickers_years]
            )

        # Check if any ticker completely failed (no files found even with fallback)
        if failed_tickers:
            logger.error(
                f"Query [{query_item.query_id}] failed: No files found for ticker(s): {failed_tickers}. "
                f"Tried original years and fallback to default year {default_year}."
            )
            result.final_answer = (
                f"Cannot process query: No financial report files found for company/companies: {', '.join(failed_tickers)}. "
                f"Tried both the requested years and fallback to default year {default_year}."
            )
            result.error = f"No files found for tickers: {failed_tickers}"
            return result

        # Check if we have any valid files at all
        if not valid_paths:
            logger.warning(
                f"No valid files found for query [{query_item.query_id}]."
            )
            result.final_answer = "No valid financial report files (PDF or TXT) found for the specified companies and years."
            result.error = "All files missing"
            return result

        logger.info(
            f"Found {len(valid_paths)} valid file(s) for query [{query_item.query_id}]: {valid_paths}"
        )

        # Step 4: Build messages with PDF files for LLM
        # Note: This implementation uses the file paths approach
        # The actual implementation depends on the API's capability to handle PDF files
        # For OpenAI-compatible APIs that support file attachments, we encode as base64

        user_prompt = FILE_BASED_ANSWER_USER_PROMPT.format(
            query=query_item.query
        )

        # Build message content with file references
        # For APIs that support multimodal input (like GPT-4V), we can include files directly
        # For text-only APIs, we would need to extract text from PDFs first

        # Check if we should use multimodal approach (sending PDF as base64)
        # This depends on the model capabilities
        messages = [
            {"role": "system", "content": FILE_BASED_ANSWER_SYSTEM_PROMPT},
        ]

        # Build user message with file attachments
        # For OpenAI's API format with file input capability
        user_content = []

        # Add text prompt
        user_content.append({"type": "text", "text": user_prompt})

        # Add document files as file type content (supports PDF and TXT)
        for file_path in valid_paths:
            file_name = os.path.basename(file_path)
            file_base64 = encode_file_to_base64(file_path)
            mime_type = get_mime_type(file_path)
            user_content.append(
                {
                    "type": "file",
                    "file": {
                        "filename": file_name,
                        "file_data": f"data:{mime_type};base64,{file_base64}",
                    },
                }
            )
            logger.info(f"Attached file: {file_name} ({mime_type})")

        messages.append({"role": "user", "content": user_content})

        # Step 5: Generate answer
        logger.info(
            f"Generating answer for query [{query_item.query_id}] with {len(valid_paths)} files..."
        )

        # Debug log sanitized messages (without base64 data)
        logger.debug(
            f"Sending messages (sanitized): {sanitize_messages_for_logging(messages)}"
        )

        final_answer = await inference_facade.create_chat_completion_async(
            messages
        )
        result.final_answer = final_answer.strip()

        logger.info(f"✓ Completed query [{query_item.query_id}]")

    except Exception as e:
        # Log error without including the full messages (which may contain large base64 data)
        logger.error(
            f"✗ Error processing query [{query_item.query_id}]: {str(e)}"
        )
        logger.debug(
            f"Error details for query [{query_item.query_id}]", exc_info=True
        )
        result.error = str(e)
        result.final_answer = f"Error: {str(e)}"

    return result


async def process_queries_concurrently(
    query_items: List[QueryItem],
    data_dir: str,
    default_year: int,
    rewriting_service: RewritingService,
    inference_facade: InferenceFacade,
    max_concurrent_tasks: int = 3,
    start_row: Optional[int] = None,
    end_row: Optional[int] = None,
    csv_writer: Optional[StreamingCSVWriter] = None,
) -> List[QueryResult]:
    """
    Process multiple queries concurrently.

    Args:
        query_items: List of queries to process
        data_dir: Base directory containing financial report documents (PDF or TXT)
        default_year: Default year for queries without explicit year
        rewriting_service: Service for extracting ticker info
        inference_facade: Service for LLM inference
        max_concurrent_tasks: Maximum number of concurrent tasks
        start_row: Starting row number (1-based) for progress tracking
        end_row: Ending row number (1-based) for progress tracking
        csv_writer: Optional StreamingCSVWriter for incremental output

    Returns:
        List of query results in the same order as input
    """
    # Create semaphore to limit concurrent tasks
    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    # Store results in a list with None placeholders to maintain order
    results_list: List[Optional[QueryResult]] = [None] * len(query_items)

    async def process_with_semaphore(
        query_item: QueryItem, row_index: int
    ) -> None:
        async with semaphore:
            # Calculate progress info and original row if start/end rows are provided
            progress_info = None
            original_row = row_index + 1  # Default to 1-based index
            if start_row is not None and end_row is not None:
                original_row = start_row + row_index
                progress_info = (original_row, start_row, end_row)

            try:
                result = await process_single_query(
                    query_item,
                    data_dir,
                    default_year,
                    rewriting_service,
                    inference_facade,
                    progress_info,
                )
            except Exception as e:
                logger.error(f"Exception in query processing: {e}")
                result = QueryResult(
                    query_item=query_item,
                    error=str(e),
                    final_answer=f"Error: {str(e)}",
                )

            # Store result in the correct position
            results_list[row_index] = result

            # Write to CSV immediately if writer is provided
            if csv_writer:
                await csv_writer.write_result(result, original_row)

    # Create tasks for all queries with their indices
    tasks = [
        process_with_semaphore(query_item, i)
        for i, query_item in enumerate(query_items)
    ]

    # Run all tasks concurrently
    await asyncio.gather(*tasks, return_exceptions=False)

    # Return results (all should be populated now)
    return [r for r in results_list if r is not None]


async def main():
    """Main entry point for the filename-based query processing pipeline."""
    # Parse command-line arguments
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("=" * 80)
    print("FILENAME-BASED FINANCIAL REPORT QUERY PIPELINE")
    print("=" * 80)

    # Validate data directory
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory does not exist: {args.data_dir}")
        sys.exit(1)

    if not os.path.isdir(args.data_dir):
        logger.error(f"Data path is not a directory: {args.data_dir}")
        sys.exit(1)

    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Default year: {args.default_year}")

    # Validate arguments for CSV/HF input
    if (args.csv or args.hf_dataset) and not args.column:
        logger.error("--column is required when using --csv or --hf-dataset")
        sys.exit(1)

    # Create input source
    try:
        input_source = create_input_source(args)
        logger.info(f"Input source created: {type(input_source).__name__}")
    except ValueError as e:
        logger.error(str(e))
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

    total_loaded = len(query_items)
    logger.info(f"Loaded {total_loaded} queries from source")

    # Apply row range filtering (only for CSV and HuggingFace datasets)
    start_row = args.start_row
    end_row = args.end_row

    if args.query:
        # CLI query - ignore row range arguments
        if start_row is not None or end_row is not None:
            logger.warning(
                "--start-row and --end-row are ignored for CLI queries"
            )
        start_row = 1
        end_row = 1
    else:
        # CSV or HuggingFace dataset - apply row filtering
        # Default start_row to 1 if not specified
        if start_row is None:
            start_row = 1

        # Default end_row to total rows if not specified
        if end_row is None:
            end_row = total_loaded

        # Validate row range
        if start_row < 1:
            logger.error(f"--start-row must be >= 1, got {start_row}")
            sys.exit(1)

        if end_row < start_row:
            logger.error(
                f"--end-row ({end_row}) must be >= --start-row ({start_row})"
            )
            sys.exit(1)

        if start_row > total_loaded:
            logger.error(
                f"--start-row ({start_row}) exceeds total rows ({total_loaded})"
            )
            sys.exit(1)

        # Clamp end_row to total rows
        if end_row > total_loaded:
            logger.warning(
                f"--end-row ({end_row}) exceeds total rows ({total_loaded}), clamping to {total_loaded}"
            )
            end_row = total_loaded

        # Apply filtering (convert 1-based to 0-based indexing)
        query_items = query_items[start_row - 1 : end_row]
        logger.info(
            f"Filtered to rows {start_row}-{end_row}: {len(query_items)} queries"
        )

    logger.info(
        f"Processing {len(query_items)} queries (rows {start_row} to {end_row})"
    )

    # Initialize inference facade
    inference_facade = InferenceFacade(model=args.model, base_url=args.base_url)

    # Initialize rewriting service (for ticker extraction)
    rewriting_service = RewritingService(inference_facade)

    # Determine output CSV path (auto-generate if flag provided without path)
    output_csv_path = None
    if (
        args.output_csv is not None
    ):  # Flag was provided (either with or without path)
        if args.output_csv == "":
            # Auto-generate filename
            # Determine source name (CSV filename or HuggingFace dataset ID)
            if args.csv:
                source_name = Path(
                    args.csv
                ).stem  # Get filename without extension
            elif args.hf_dataset:
                # Replace slashes with underscores for HF dataset IDs like "org/dataset"
                source_name = args.hf_dataset.replace("/", "_")
            else:
                source_name = "cli_query"

            # Sanitize model name for filename (replace slashes and other invalid chars)
            model_name = (
                args.model.replace("/", "_").replace(":", "_").replace(" ", "_")
            )

            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Build filename
            filename = f"{source_name}-{start_row}-to-{end_row}-{model_name}-{timestamp}.csv"

            # Ensure output directory exists
            output_dir = Path("./output")
            output_dir.mkdir(parents=True, exist_ok=True)

            output_csv_path = str(output_dir / filename)
            logger.info(f"Auto-generated output path: {output_csv_path}")
        else:
            output_csv_path = args.output_csv

    # Initialize streaming CSV writer if output path is specified
    csv_writer = None
    fieldnames = [
        "original_row",
        "query_id",
        "query",
        "tickers",
        "file_paths",
        "missing_files",
        "final_answer",
        "error",
    ]

    if output_csv_path:
        csv_writer = StreamingCSVWriter(output_csv_path, fieldnames)
        await csv_writer.initialize()
        logger.info(f"Streaming results to: {output_csv_path}")

    # Process queries concurrently
    logger.info(
        f"\nProcessing {len(query_items)} queries with {args.concurrent} concurrent tasks..."
    )
    print(
        f"\nProcessing {len(query_items)} queries (rows {start_row} to {end_row})..."
    )

    results = await process_queries_concurrently(
        query_items,
        args.data_dir,
        args.default_year,
        rewriting_service,
        inference_facade,
        max_concurrent_tasks=args.concurrent,
        start_row=start_row,
        end_row=end_row,
        csv_writer=csv_writer,
    )

    # Output results to terminal if no CSV output specified
    if not output_csv_path:
        output_to_terminal(results)

    # Summary
    successful = sum(1 for r in results if not r.error)
    failed = len(results) - successful

    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Row range:      {start_row} to {end_row}")
    print(f"Total queries:  {len(results)}")
    print(f"Successful:     {successful}")
    print(f"Failed:         {failed}")
    if output_csv_path:
        print(f"Output file:    {output_csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
