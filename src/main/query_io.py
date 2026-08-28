#!/usr/bin/env python
"""
Shared input/output handling for query processing pipelines.

This module provides common classes for:
- Reading queries from multiple sources (CLI, CSV, HuggingFace)
- Outputting results to terminal or CSV
"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


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
    final_answer: str = ""
    error: Optional[str] = None
    # Additional metadata that can be extended by specific pipelines
    metadata: dict = field(default_factory=dict)


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


def output_to_terminal(results: List[QueryResult]):
    """Output results to terminal."""
    print("\n" + "=" * 80)
    print("QUERY RESULTS")
    print("=" * 80)
    
    for i, result in enumerate(results, 1):
        print(f"\n[Query {i}] ID: {result.query_item.query_id}")
        print(f"Question: {result.query_item.query}")
        
        # Print metadata if available
        if result.metadata:
            for key, value in result.metadata.items():
                print(f"{key}: {value}")
        
        print(f"\nAnswer:\n{result.final_answer}")
        print("-" * 80)


def output_to_csv(results: List[QueryResult], output_path: str, fieldnames: Optional[List[str]] = None):
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
        
        # Default fieldnames
        if fieldnames is None:
            fieldnames = ['query_id', 'query', 'final_answer']
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                # Format query and answer as single-line strings by replacing newlines
                formatted_query = result.query_item.query.replace('\n', ' ').replace('\r', ' ')
                formatted_query = ' '.join(formatted_query.split())
                
                formatted_answer = result.final_answer.replace('\n', ' ').replace('\r', ' ')
                formatted_answer = ' '.join(formatted_answer.split())
                
                row = {
                    'query_id': result.query_item.query_id,
                    'query': formatted_query,
                    'final_answer': formatted_answer
                }
                
                # Add any metadata fields that are in fieldnames
                for key in fieldnames:
                    if key not in row and key in result.metadata:
                        value = result.metadata[key]
                        if isinstance(value, str):
                            value = value.replace('\n', ' ').replace('\r', ' ')
                            value = ' '.join(value.split())
                        row[key] = value
                
                writer.writerow(row)
        
        logger.info(f"Results written to CSV file: {output_path}")
        print(f"\n✓ Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error writing to CSV file: {e}")
        raise


def create_input_source(args) -> QueryInputSource:
    """
    Create an input source based on parsed arguments.
    
    Args:
        args: Parsed argparse namespace with query, csv, hf_dataset, column, id_column, split
        
    Returns:
        Configured QueryInputSource instance
    """
    if args.query:
        return CLIInputSource(args.query)
    elif args.csv:
        if not args.column:
            raise ValueError("--column is required when using --csv")
        return CSVInputSource(args.csv, args.column, args.id_column)
    elif args.hf_dataset:
        if not args.column:
            raise ValueError("--column is required when using --hf-dataset")
        return HuggingFaceInputSource(
            args.hf_dataset, 
            args.column, 
            args.split,
            args.id_column
        )
    else:
        raise ValueError("No input source specified")
