"""
Script to get random rows from a Hugging Face dataset and save to CSV.

Usage:
    python scripts/get_random_queries_to_csv.py --dataset_id <dataset_id> --total <number> --id_field <id_column> --query_field <query_column>

Example:
    python scripts/get_random_queries_to_csv.py --dataset_id "ms_marco" --total 100 --id_field "query_id" --query_field "query"
"""

import argparse
import os
import random
from datetime import datetime

import pandas as pd
from datasets import load_dataset


def get_random_queries(
    dataset_id: str,
    total_queries: int,
    id_field: str,
    query_field: str,
    split: str = "train",
    subset: str | None = None,
) -> pd.DataFrame:
    """
    Load a Hugging Face dataset and randomly sample rows.

    Args:
        dataset_id: The Hugging Face dataset identifier (e.g., "ms_marco")
        total_queries: Number of random queries to select
        id_field: Column name for the query ID
        query_field: Column name for the query text
        split: Dataset split to use (default: "train")
        subset: Optional dataset subset/configuration name

    Returns:
        DataFrame with query_id and query columns
    """
    print(f"Loading dataset: {dataset_id}")
    if subset:
        dataset = load_dataset(dataset_id, subset, split=split)
    else:
        dataset = load_dataset(dataset_id, split=split)

    print(f"Dataset loaded. Total rows: {len(dataset)}")

    # Validate fields exist
    if id_field not in dataset.column_names:
        raise ValueError(
            f"ID field '{id_field}' not found in dataset. "
            f"Available columns: {dataset.column_names}"
        )
    if query_field not in dataset.column_names:
        raise ValueError(
            f"Query field '{query_field}' not found in dataset. "
            f"Available columns: {dataset.column_names}"
        )

    # Ensure we don't request more than available
    actual_total = min(total_queries, len(dataset))
    if actual_total < total_queries:
        print(
            f"Warning: Requested {total_queries} queries but dataset only has {len(dataset)}. "
            f"Using {actual_total} instead."
        )

    # Randomly sample indices
    indices = random.sample(range(len(dataset)), actual_total)
    sampled_data = dataset.select(indices)

    # Create DataFrame with standardized column names
    df = pd.DataFrame(
        {
            "query_id": sampled_data[id_field],
            "query": sampled_data[query_field],
        }
    )

    return df


def save_to_csv(
    df: pd.DataFrame, dataset_id: str, total: int, output_dir: str
) -> str:
    """
    Save DataFrame to CSV with formatted filename.

    Args:
        df: DataFrame to save
        dataset_id: Original dataset ID (used in filename)
        total: Number of queries (used in filename)
        output_dir: Directory to save the CSV file

    Returns:
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Format dataset_id for filename (replace / with _)
    safe_dataset_id = dataset_id.replace("/", "_")

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create filename
    filename = f"{safe_dataset_id}-{total}-{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)

    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"Saved {len(df)} queries to: {filepath}")

    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Get random queries from a Hugging Face dataset and save to CSV"
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        required=True,
        help="Hugging Face dataset identifier (e.g., 'ms_marco', 'squad')",
    )
    parser.add_argument(
        "--total",
        type=int,
        required=True,
        help="Total number of random queries to select",
    )
    parser.add_argument(
        "--id_field",
        type=str,
        required=True,
        help="Column name in the dataset that contains the query ID",
    )
    parser.add_argument(
        "--query_field",
        type=str,
        required=True,
        help="Column name in the dataset that contains the query text",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Optional dataset subset/configuration name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./random_queries_csv",
        help="Output directory for the CSV file (default: ./random_queries_csv)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    # Get random queries
    df = get_random_queries(
        dataset_id=args.dataset_id,
        total_queries=args.total,
        id_field=args.id_field,
        query_field=args.query_field,
        split=args.split,
        subset=args.subset,
    )

    # Save to CSV
    save_to_csv(
        df=df,
        dataset_id=args.dataset_id,
        total=args.total,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
