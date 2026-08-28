#!/usr/bin/env python
"""
Concurrent document indexing pipeline.

This script:
1. Takes a folder path as input from the terminal
2. Finds all supported files in the folder (recursively)
3. Uses ProcessPoolExecutor to process documents in parallel (parse + chunk)
4. Uses async logic to calculate embeddings and add to vector DB
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, cpu_count

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from beans.embedder import Embedder
from beans.faiss_manager import FaissIndex
from beans.sqlite3_db_manager import Sqlite3DbManager
from dao.document_dao import DocumentDao
from dao.chunk_dao import ChunkDao
from service.document_processor import DocumentProcessor
from enums.file_type import FileType
from data_classes.embedding_generator_config import EmbedderConfig
from constants import DEFAULT_EMBEDDING_DIM, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default database and index file names (in project root)
DEFAULT_DB_NAME = "rag_pipeline.db"
DEFAULT_FAISS_INDEX_NAME = "faiss_index.index"


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


def find_supported_files(folder_path: str) -> List[str]:
    """
    Recursively find all supported files in the given folder.
    
    Args:
        folder_path: Path to the folder to search
        
    Returns:
        List of absolute file paths for supported files
    """
    supported_files = []
    folder = Path(folder_path)
    
    if not folder.exists():
        logger.error(f"Folder does not exist: {folder_path}")
        return []
    
    if not folder.is_dir():
        logger.error(f"Path is not a directory: {folder_path}")
        return []
    
    # Get all supported extensions
    supported_extensions = {ft.value for ft in FileType if ft != FileType.OTHER}
    logger.info(f"Supported file types: {supported_extensions}")
    
    # Walk through directory recursively
    for file_path in folder.rglob("*"):
        if file_path.is_file():
            extension = file_path.suffix.lstrip('.').lower()
            if extension in supported_extensions:
                supported_files.append(str(file_path.absolute()))
    
    return supported_files


def process_single_document(args: Tuple[str, bool, str, str]) -> Optional[Tuple[int, str]]:
    """
    Process a single document in a separate process.
    This function is designed to be called by ProcessPoolExecutor.
    
    Args:
        args: Tuple of (file_path, overwrite, db_path, faiss_index_path)
        
    Returns:
        Tuple of (document_id, file_path) if successful, None otherwise
    """
    file_path, overwrite, db_path, faiss_index_path = args
    
    try:
        # Set up logging for this process
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        process_logger = logging.getLogger(__name__)
        
        # Create database connection (each process needs its own)
        # Use Sqlite3DbManager for proper WAL mode and timeout settings
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        schema_path = str(project_root / "schemas" / "fts_schema.sql")
        
        db_manager = Sqlite3DbManager(
            db_path=db_path,
            schema_file=schema_path
        )
        Session = db_manager.session_factory
        
        # Initialize DAOs
        document_dao = DocumentDao(Session)
        chunk_dao = ChunkDao(Session)
        
        # Initialize embedder (just for compatibility, not used in this stage)
        embedder_config = EmbedderConfig()
        embedder = Embedder(embedder_config)
        
        # Create document processor (no FAISS index needed for parse/chunk stage)
        processor = DocumentProcessor(
            document_dao=document_dao,
            chunk_dao=chunk_dao,
            embedder=embedder,
            faiss_index=None,  # Not needed for parsing and chunking
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP
        )
        
        # Process the document (parse + chunk + save to DB)
        document = processor.process_document(file_path, overwrite=overwrite)
        
        if document and document.id:
            process_logger.info(f"✓ Processed: {os.path.basename(file_path)} (ID: {document.id})")
            return (document.id, file_path)
        else:
            process_logger.error(f"✗ Failed to process: {file_path}")
            return None
            
    except Exception as e:
        process_logger = logging.getLogger(__name__)
        process_logger.error(f"✗ Error processing {file_path}: {str(e)}", exc_info=True)
        return None


async def calculate_embeddings_for_document(
    document_id: int,
    file_path: str,
    embedder: Embedder,
    chunk_dao: ChunkDao,
    faiss_index: FaissIndex
) -> bool:
    """
    Asynchronously calculate embeddings for a document and add to FAISS.
    
    Args:
        document_id: ID of the document to process
        file_path: Path to the document file (for logging)
        embedder: Embedder instance with async support
        chunk_dao: ChunkDao instance
        faiss_index: FaissIndex instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"[Async] Generating embeddings for document {document_id}")
        
        # Fetch chunks for the document
        chunks = chunk_dao.get_chunks_by_document_id(document_id)
        
        if len(chunks) == 0:
            logger.warning(f"[Async] No chunks found for document {document_id}")
            return False
        
        # Extract chunk IDs and texts
        chunk_ids = [chunk.id for chunk in chunks]
        chunk_texts = [chunk.chunk_text for chunk in chunks]
        
        # Generate embeddings asynchronously
        logger.info(f"[Async] Generating embeddings for {len(chunk_texts)} chunks...")
        import numpy as np
        embedding_vectors = await embedder.generate_embeddings_async(chunk_texts)
        embedding_ids = np.array(chunk_ids, dtype=np.int64)
        
        # Add embeddings to FAISS index (this is synchronized with lock)
        logger.info(f"[Async] Adding embeddings to FAISS index...")
        completed_chunk_ids = faiss_index.add_embeddings(embedding_vectors, embedding_ids)
        
        logger.info(f"✓ [Async] Completed embeddings for {os.path.basename(file_path)} "
                   f"({len(completed_chunk_ids)} chunks)")
        return True
        
    except Exception as e:
        logger.error(f"✗ [Async] Error calculating embeddings for document {document_id}: {str(e)}", 
                    exc_info=True)
        return False


async def process_embeddings_concurrently(
    document_results: List[Tuple[int, str]],
    db_path: str,
    faiss_index_path: str,
    max_concurrent_tasks: int = 5
) -> Tuple[int, int]:
    """
    Process embeddings for multiple documents concurrently using asyncio.
    
    Args:
        document_results: List of (document_id, file_path) tuples
        db_path: Path to the SQLite database
        faiss_index_path: Path to the FAISS index file
        max_concurrent_tasks: Maximum number of concurrent async tasks
        
    Returns:
        Tuple of (success_count, failure_count)
    """
    # Create database connection for async operations
    # Use Sqlite3DbManager for proper WAL mode and timeout settings
    db_manager = Sqlite3DbManager(
        db_path=db_path,
        schema_file=str(project_root / "schemas" / "fts_schema.sql")
    )
    Session = db_manager.session_factory
    
    # Initialize DAOs
    chunk_dao = ChunkDao(Session)
    
    # Initialize embedder with async support
    embedder_config = EmbedderConfig()
    embedder = Embedder(embedder_config)
    
    # Initialize FAISS index with lock (using multiprocessing Manager for lock)
    with Manager() as manager:
        faiss_lock = manager.Lock()
        faiss_index = FaissIndex(
            index_file=faiss_index_path,
            dimension=DEFAULT_EMBEDDING_DIM,
            faiss_lock=faiss_lock
        )
        
        # Load or create FAISS index
        faiss_index.load_or_build_from_scratch()
        
        # Create semaphore to limit concurrent tasks
        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        async def process_with_semaphore(doc_id: int, file_path: str):
            async with semaphore:
                return await calculate_embeddings_for_document(
                    doc_id, file_path, embedder, chunk_dao, faiss_index
                )
        
        # Create tasks for all documents
        tasks = [
            process_with_semaphore(doc_id, file_path) 
            for doc_id, file_path in document_results
        ]
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successes and failures
        success_count = sum(1 for r in results if r is True)
        failure_count = len(results) - success_count
        
        return success_count, failure_count


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='RAG Pipeline - Concurrent Document Indexing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Index a folder
  python main/main_index.py /path/to/documents
  
  # Index with a custom project name
  python main/main_index.py /path/to/documents --project-name myproject
  
  # Index with overwrite
  python main/main_index.py /path/to/documents --overwrite
  
  # Adjust concurrency
  python main/main_index.py /path/to/documents --workers 4 --async-tasks 10
  
  # Interactive mode (prompts for folder)
  python main/main_index.py

Pipeline Stages:
  Stage 1: Parse and chunk documents in parallel (CPU-bound)
  Stage 2: Generate embeddings asynchronously (I/O-bound)

Output:
  • rag_pipeline.db - SQLite database with documents and chunks (default)
  • faiss_index.index - FAISS vector index for similarity search (default)
  • Or {project_name}.rag_pipeline.db and {project_name}.faiss_index.index if --project-name is used

For more information, see main/README.md
        '''
    )
    
    parser.add_argument(
        'folder',
        nargs='?',
        help='Path to folder containing documents to index (if not provided, will prompt)'
    )
    
    parser.add_argument(
        '-p', '--project-name',
        type=str,
        default=None,
        metavar='NAME',
        help='Project name for database and index files (e.g., "myproject" creates myproject.rag_pipeline.db and myproject.faiss_index.index)'
    )
    
    parser.add_argument(
        '-o', '--overwrite',
        action='store_true',
        help='Overwrite existing documents in the database'
    )
    
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=None,
        metavar='N',
        help='Number of worker processes for Stage 1 (default: CPU count)'
    )
    
    parser.add_argument(
        '-a', '--async-tasks',
        type=int,
        default=5,
        metavar='N',
        help='Number of concurrent async tasks for Stage 2 (default: 5)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the indexing pipeline."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("=" * 80)
    print("RAG PIPELINE - CONCURRENT DOCUMENT INDEXING")
    print("=" * 80)
    
    # Get folder path from arguments or prompt user
    if args.folder:
        folder_path = args.folder
    else:
        folder_path = input("\nEnter folder path to index: ").strip()
    
    if not folder_path:
        logger.error("No folder path provided. Exiting.")
        return
    
    # Convert to absolute path
    folder_path = os.path.abspath(folder_path)
    logger.info(f"Indexing folder: {folder_path}")
    
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
    
    # Find all supported files
    logger.info("Searching for supported files...")
    supported_files = find_supported_files(folder_path)
    
    if not supported_files:
        logger.warning("No supported files found in the folder.")
        return
    
    logger.info(f"Found {len(supported_files)} supported file(s)")
    for i, file in enumerate(supported_files[:5], 1):
        logger.info(f"  {i}. {os.path.basename(file)}")
    if len(supported_files) > 5:
        logger.info(f"  ... and {len(supported_files) - 5} more")
    
    # Use overwrite from arguments or prompt user
    if args.overwrite:
        overwrite = True
        logger.info("\nOverwrite mode: ENABLED (from --overwrite flag)")
    else:
        overwrite_input = input("\nOverwrite existing documents? (y/n, default: n): ").strip().lower()
        overwrite = overwrite_input == 'y'
    
    # Set up database using Sqlite3DbManager
    logger.info(f"\nSetting up database: {db_path}")
    schema_path = str(project_root / "schemas" / "fts_schema.sql")
    
    db_manager = Sqlite3DbManager(
        db_path=db_path,
        schema_file=schema_path
    )
    
    # Get the session factory for DAOs
    Session = db_manager.session_factory
    logger.info("Database setup complete (WAL mode enabled for concurrent access)")
    
    # ============================================================================
    # STAGE 1: Parallel document processing (parse + chunk + save to DB)
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 1: Processing documents (parse + chunk)")
    logger.info("=" * 80)
    
    # Determine number of workers
    if args.workers:
        num_workers = min(args.workers, len(supported_files))
        logger.info(f"Using {num_workers} worker processes (from --workers flag)")
    else:
        num_workers = min(cpu_count(), len(supported_files))
        logger.info(f"Using {num_workers} worker processes (CPU count: {cpu_count()})")
    
    document_results = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_document, (file_path, overwrite, db_path, faiss_index_path)): file_path
            for file_path in supported_files
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_file):
            result = future.result()
            if result:
                document_results.append(result)
    
    logger.info(f"\nStage 1 complete: {len(document_results)}/{len(supported_files)} documents processed")
    
    if not document_results:
        logger.warning("No documents were successfully processed. Exiting.")
        return
    
    # ============================================================================
    # STAGE 2: Async embedding calculation and FAISS indexing
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 2: Calculating embeddings and building vector index")
    logger.info("=" * 80)
    
    # Run async embedding processing
    logger.info(f"Using {args.async_tasks} concurrent async tasks")
    success_count, failure_count = asyncio.run(
        process_embeddings_concurrently(
            document_results, 
            db_path=db_path,
            faiss_index_path=faiss_index_path,
            max_concurrent_tasks=args.async_tasks
        )
    )
    
    logger.info(f"\nStage 2 complete: {success_count} successful, {failure_count} failed")
    
    # ============================================================================
    # Summary
    # ============================================================================
    print("\n" + "=" * 80)
    print("INDEXING COMPLETE")
    print("=" * 80)
    print(f"Documents found:      {len(supported_files)}")
    print(f"Documents processed:  {len(document_results)}")
    print(f"Embeddings generated: {success_count}")
    print(f"Embeddings failed:    {failure_count}")
    print(f"\nDatabase: {db_path}")
    print(f"FAISS Index: {faiss_index_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

