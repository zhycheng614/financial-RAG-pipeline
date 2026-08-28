#!/usr/bin/env python
"""
Benchmark pipeline for evaluating RAG system results against ground truth.

This script:
1. Takes RAG results from a CSV file with columns:
   ["original_row", "query_id", "query", "tickers", "file_paths", "missing_files", "final_answer", "error"]
2. Loads ground truth from a HuggingFace dataset
3. Uses OpenAI API to evaluate each RAG answer against the ground truth
4. Outputs evaluation scores to a CSV file (streaming)
5. Prints the final average score

Scoring: 1 (totally wrong) to 10 (totally correct)
- 1: Completely incorrect answer
- 5: 50% correct/missing information
- 10: Fully correct answer
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from beans.inference_facade import InferenceFacade
from beans.anthropic_inference_facade import AnthropicInferenceFacade
from data_classes.inference_parameters import InferenceParameters
from prompts import (BENCHMARK_EVALUATION_SYSTEM_PROMPT,
                     BENCHMARK_EVALUATION_USER_PROMPT)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default settings
DEFAULT_MODEL = "gpt-4.1"
DEFAULT_BASE_URL = "https://api.openai.com/v1"


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
                with open(self.filepath, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(
                        f, 
                        fieldnames=self.fieldnames,
                        quoting=csv.QUOTE_ALL
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
        """
        if value is None:
            return ''
        str_value = str(value)
        
        # Replace actual newlines with literal \n
        str_value = str_value.replace('\r\n', '\\n').replace('\n', '\\n').replace('\r', '\\n')
        
        # Convert Unicode special characters to ASCII equivalents
        unicode_replacements = {
            '\u2018': "'", '\u2019': "'", '\u201C': '"', '\u201D': '"',
            '\u2013': '-', '\u2014': '--', '\u00D7': 'x', '\u00F7': '/',
            '\u2212': '-', '\u00B1': '+/-', '\u2026': '...', '\u2022': '-',
            '\u00B0': ' deg', '\u00A0': ' ',
        }
        
        for unicode_char, ascii_equiv in unicode_replacements.items():
            str_value = str_value.replace(unicode_char, ascii_equiv)
        
        return str_value
    
    async def write_result(self, row_data: Dict[str, Any]):
        """
        Write a single result row to the CSV file.
        
        Args:
            row_data: Dictionary with the row data to write
        """
        async with self.lock:
            if not self._initialized:
                raise RuntimeError("StreamingCSVWriter not initialized. Call initialize() first.")
            
            # Sanitize all text values
            sanitized_row = {
                k: self._sanitize_text(v) if k in self.fieldnames else v 
                for k, v in row_data.items()
            }
            
            # Filter to only include fields in fieldnames
            sanitized_row = {k: v for k, v in sanitized_row.items() if k in self.fieldnames}
            
            # Append to CSV with QUOTE_ALL
            with open(self.filepath, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(
                    f, 
                    fieldnames=self.fieldnames,
                    quoting=csv.QUOTE_ALL
                )
                writer.writerow(sanitized_row)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Benchmark RAG Results Against Ground Truth',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic usage with HuggingFace dataset
  python main/main_benchmark_rag_result.py \\
      --input-csv ./output/rag_results.csv \\
      --hf-dataset "org/dataset_name" \\
      --gt-field "answer" \\
      --id-field "id"

  # With custom output path
  python main/main_benchmark_rag_result.py \\
      --input-csv ./output/rag_results.csv \\
      --hf-dataset "org/dataset_name" \\
      --gt-field "answer" \\
      --id-field "id" \\
      --output-csv ./output/benchmark_results.csv

  # With custom concurrency and model
  python main/main_benchmark_rag_result.py \\
      --input-csv ./output/rag_results.csv \\
      --hf-dataset "org/dataset_name" \\
      --gt-field "answer" \\
      --id-field "id" \\
      --concurrent 5 \\
      --model gpt-4o

Input CSV Expected Columns:
  original_row, query_id, query, tickers, file_paths, missing_files, final_answer, error

Output CSV Columns:
  original_row, query_id, query, rag_answer, ground_truth, score, evaluation_error
        '''
    )
    
    # Required arguments
    parser.add_argument(
        '--input-csv',
        type=str,
        required=True,
        metavar='PATH',
        help='Path to the RAG results CSV file'
    )
    parser.add_argument(
        '--hf-dataset',
        type=str,
        required=True,
        metavar='DATASET_ID',
        help='HuggingFace dataset ID containing ground truth'
    )
    parser.add_argument(
        '--gt-field',
        type=str,
        required=True,
        metavar='FIELD_NAME',
        help='Field name in the HuggingFace dataset that contains the ground truth answer'
    )
    parser.add_argument(
        '--id-field',
        type=str,
        required=True,
        metavar='FIELD_NAME',
        help='Field name in the HuggingFace dataset to match with query_id from CSV'
    )
    
    # Optional arguments
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        metavar='NAME',
        help='Dataset split to use (default: train)'
    )
    parser.add_argument(
        '--output-csv',
        type=str,
        nargs='?',
        const='',
        default=None,
        metavar='PATH',
        help='Output CSV path. If flag provided without path, auto-generates filename'
    )
    parser.add_argument(
        '--concurrent',
        type=int,
        default=3,
        metavar='N',
        help='Number of concurrent evaluation tasks (default: 3)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL,
        metavar='NAME',
        help=f'Model name for evaluation (default: {DEFAULT_MODEL})'
    )
    parser.add_argument(
        '--base-url',
        type=str,
        default=DEFAULT_BASE_URL,
        metavar='URL',
        help=f'Base URL for inference service (default: {DEFAULT_BASE_URL})'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )
    
    return parser.parse_args()


def load_rag_results(csv_path: str) -> List[Dict[str, str]]:
    """
    Load RAG results from a CSV file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        List of dictionaries, each representing a row
    """
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(dict(row))
    return results


def load_ground_truth_dataset(dataset_id: str, split: str, id_field: str, gt_field: str) -> Dict[str, str]:
    """
    Load ground truth from a HuggingFace dataset and build a lookup dictionary.
    
    Args:
        dataset_id: HuggingFace dataset identifier
        split: Dataset split to use
        id_field: Field name to use as the key (matches query_id)
        gt_field: Field name containing the ground truth answer
        
    Returns:
        Dictionary mapping id_field values to ground truth answers
    """
    from datasets import load_dataset
    
    logger.info(f"Loading HuggingFace dataset: {dataset_id} (split: {split})")
    dataset = load_dataset(dataset_id, split=split)
    
    # Build lookup dictionary
    ground_truth_lookup = {}
    for item in dataset:
        key = str(item[id_field])
        value = str(item[gt_field])
        ground_truth_lookup[key] = value
    
    logger.info(f"Loaded {len(ground_truth_lookup)} ground truth entries")
    return ground_truth_lookup


async def evaluate_single_answer(
    row_data: Dict[str, str],
    ground_truth: str,
    inference_facade: InferenceFacade,
    row_index: int,
    total_rows: int
) -> Dict[str, Any]:
    """
    Evaluate a single RAG answer against the ground truth.
    
    Args:
        row_data: Dictionary containing the RAG result row
        ground_truth: The ground truth answer
        inference_facade: InferenceFacade for LLM calls
        row_index: Current row index (for progress logging)
        total_rows: Total number of rows (for progress logging)
        
    Returns:
        Dictionary with evaluation results
    """
    query_id = row_data.get('query_id', '')
    query = row_data.get('query', '')
    rag_answer = row_data.get('final_answer', '')
    original_row = row_data.get('original_row', str(row_index + 1))
    
    result = {
        'original_row': original_row,
        'query_id': query_id,
        'query': query,
        'rag_answer': rag_answer,
        'ground_truth': ground_truth,
        'score': None,
        'evaluation_error': None
    }
    
    try:
        logger.info(f"[{row_index + 1}/{total_rows}] Evaluating query_id: {query_id}")
        
        # Check if RAG answer has an error
        rag_error = row_data.get('error', '')
        if rag_error and rag_error.strip():
            logger.warning(f"RAG result has error for query_id {query_id}: {rag_error}")
            # Still evaluate - the LLM can assess if the error response is reasonable
        
        # Build evaluation prompt
        user_prompt = BENCHMARK_EVALUATION_USER_PROMPT.format(
            query=query,
            ground_truth=ground_truth,
            rag_answer=rag_answer
        )
        
        messages = [
            {"role": "system", "content": BENCHMARK_EVALUATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        # Configure inference parameters for JSON output
        inference_params = InferenceParameters(
            temperature=0.0,  # Deterministic evaluation
            max_tokens=100,
            response_format={"type": "json_object"}
        )
        
        # Call LLM for evaluation
        response = await inference_facade.create_chat_completion_async(
            messages,
            inference_parameters=inference_params
        )
        
        # Parse JSON response
        try:
            response_json = json.loads(response.strip())
            score = response_json.get('score')
            
            if score is None:
                raise ValueError("No 'score' field in response")
            
            score = int(score)
            if not 1 <= score <= 10:
                raise ValueError(f"Score {score} out of range [1, 10]")
            
            result['score'] = score
            logger.info(f"[{row_index + 1}/{total_rows}] query_id {query_id}: Score = {score}")
            
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.error(f"Failed to parse evaluation response for query_id {query_id}: {e}")
            logger.debug(f"Raw response: {response}")
            result['evaluation_error'] = f"Parse error: {str(e)}"
            
    except Exception as e:
        logger.error(f"Error evaluating query_id {query_id}: {str(e)}")
        result['evaluation_error'] = str(e)
    
    return result


async def evaluate_all_answers(
    rag_results: List[Dict[str, str]],
    ground_truth_lookup: Dict[str, str],
    inference_facade: InferenceFacade,
    max_concurrent_tasks: int,
    csv_writer: Optional[StreamingCSVWriter]
) -> List[Dict[str, Any]]:
    """
    Evaluate all RAG answers concurrently.
    
    Args:
        rag_results: List of RAG result rows
        ground_truth_lookup: Dictionary mapping query_id to ground truth
        inference_facade: InferenceFacade for LLM calls
        max_concurrent_tasks: Maximum concurrent evaluations
        csv_writer: Optional CSV writer for streaming output
        
    Returns:
        List of evaluation results
    """
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    results_list: List[Optional[Dict[str, Any]]] = [None] * len(rag_results)
    total_rows = len(rag_results)
    
    async def evaluate_with_semaphore(row_data: Dict[str, str], row_index: int):
        async with semaphore:
            query_id = row_data.get('query_id', '')
            
            # Find ground truth for this query_id
            ground_truth = ground_truth_lookup.get(query_id)
            
            if ground_truth is None:
                logger.warning(f"No ground truth found for query_id: {query_id}")
                result = {
                    'original_row': row_data.get('original_row', str(row_index + 1)),
                    'query_id': query_id,
                    'query': row_data.get('query', ''),
                    'rag_answer': row_data.get('final_answer', ''),
                    'ground_truth': '',
                    'score': None,
                    'evaluation_error': f"No ground truth found for query_id: {query_id}"
                }
            else:
                result = await evaluate_single_answer(
                    row_data,
                    ground_truth,
                    inference_facade,
                    row_index,
                    total_rows
                )
            
            results_list[row_index] = result
            
            # Write to CSV immediately if writer is provided
            if csv_writer:
                await csv_writer.write_result(result)
    
    # Create tasks for all rows
    tasks = [
        evaluate_with_semaphore(row_data, i) 
        for i, row_data in enumerate(rag_results)
    ]
    
    # Run all tasks concurrently
    await asyncio.gather(*tasks, return_exceptions=False)
    
    return [r for r in results_list if r is not None]


def calculate_average_score(results: List[Dict[str, Any]]) -> Optional[float]:
    """
    Calculate the average score from evaluation results.
    Only includes rows with valid scores (excludes failed evaluations).
    
    Args:
        results: List of evaluation result dictionaries
        
    Returns:
        Average score or None if no valid scores
    """
    valid_scores = [r['score'] for r in results if r.get('score') is not None]
    
    if not valid_scores:
        return None
    
    return sum(valid_scores) / len(valid_scores)


async def main():
    """Main entry point for the benchmark pipeline."""
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("=" * 80)
    print("RAG BENCHMARK EVALUATION PIPELINE")
    print("=" * 80)
    
    # Validate input CSV exists
    if not os.path.exists(args.input_csv):
        logger.error(f"Input CSV file does not exist: {args.input_csv}")
        sys.exit(1)
    
    # Load RAG results
    logger.info(f"Loading RAG results from: {args.input_csv}")
    try:
        rag_results = load_rag_results(args.input_csv)
    except Exception as e:
        logger.error(f"Failed to load RAG results: {e}")
        sys.exit(1)
    
    if not rag_results:
        logger.error("No results found in input CSV")
        sys.exit(1)
    
    logger.info(f"Loaded {len(rag_results)} RAG results")
    
    # Load ground truth dataset
    try:
        ground_truth_lookup = load_ground_truth_dataset(
            args.hf_dataset,
            args.split,
            args.id_field,
            args.gt_field
        )
    except Exception as e:
        logger.error(f"Failed to load ground truth dataset: {e}")
        sys.exit(1)
    
    # Initialize inference facade — dispatch on model-name prefix.
    # claude-* models go to the Anthropic facade (Phase 5 cross-evaluator);
    # everything else stays on the OpenAI-compatible InferenceFacade.
    if args.model.lower().startswith("claude"):
        inference_facade = AnthropicInferenceFacade(model=args.model)
    else:
        inference_facade = InferenceFacade(
            model=args.model,
            base_url=args.base_url,
        )
    
    # Determine output CSV path
    output_csv_path = None
    if args.output_csv is not None:
        if args.output_csv == '':
            # Auto-generate filename
            input_csv_name = Path(args.input_csv).stem
            dataset_name = args.hf_dataset.replace('/', '_')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"benchmark-{dataset_name}-{input_csv_name}-{timestamp}.csv"
            
            output_dir = Path('./output')
            output_dir.mkdir(parents=True, exist_ok=True)
            output_csv_path = str(output_dir / filename)
            logger.info(f"Auto-generated output path: {output_csv_path}")
        else:
            output_csv_path = args.output_csv
    else:
        # Default output path
        input_csv_name = Path(args.input_csv).stem
        dataset_name = args.hf_dataset.replace('/', '_')
        filename = f"benchmark-{dataset_name}-{input_csv_name}.csv"
        
        output_dir = Path('./output')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_csv_path = str(output_dir / filename)
        logger.info(f"Using default output path: {output_csv_path}")
    
    # Initialize streaming CSV writer
    fieldnames = ['original_row', 'query_id', 'query', 'rag_answer', 'ground_truth', 'score', 'evaluation_error']
    csv_writer = StreamingCSVWriter(output_csv_path, fieldnames)
    await csv_writer.initialize()
    logger.info(f"Streaming results to: {output_csv_path}")
    
    # Run evaluations
    print(f"\nEvaluating {len(rag_results)} results with {args.concurrent} concurrent tasks...")
    print(f"Model: {args.model}")
    print(f"Ground truth dataset: {args.hf_dataset} (field: {args.gt_field})")
    print("-" * 80)
    
    results = await evaluate_all_answers(
        rag_results,
        ground_truth_lookup,
        inference_facade,
        max_concurrent_tasks=args.concurrent,
        csv_writer=csv_writer
    )
    
    # Calculate statistics
    total_evaluated = len(results)
    successful = sum(1 for r in results if r.get('score') is not None)
    failed = total_evaluated - successful
    average_score = calculate_average_score(results)
    
    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"Total evaluated:    {total_evaluated}")
    print(f"Successful:         {successful}")
    print(f"Failed:             {failed}")
    print("-" * 80)
    
    if average_score is not None:
        print(f"AVERAGE SCORE:      {average_score:.2f} / 10")
    else:
        print("AVERAGE SCORE:      N/A (no successful evaluations)")
    
    print("-" * 80)
    print(f"Output file:        {output_csv_path}")
    print("=" * 80)
    
    # Print score distribution
    if successful > 0:
        score_counts = {}
        for r in results:
            score = r.get('score')
            if score is not None:
                score_counts[score] = score_counts.get(score, 0) + 1
        
        print("\nScore Distribution:")
        for score in sorted(score_counts.keys()):
            count = score_counts[score]
            bar = "#" * count
            print(f"  {score:2d}: {count:3d} {bar}")


if __name__ == "__main__":
    asyncio.run(main())
