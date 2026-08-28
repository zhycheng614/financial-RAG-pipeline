#!/usr/bin/env python
"""
Quick script to check the status of the indexed data.
Shows statistics about documents, chunks, and FAISS index.
"""

import os
import sys
from pathlib import Path
from sqlalchemy import func

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.document import Document
from models.chunk import Chunk
from beans.faiss_manager import FaissIndex
from beans.sqlite3_db_manager import Sqlite3DbManager
from constants import DEFAULT_EMBEDDING_DIM

DB_PATH = "rag_pipeline.db"
FAISS_INDEX_PATH = "faiss_index.index"


def check_database_status():
    """Check and display database statistics."""
    print("=" * 80)
    print("DATABASE STATUS")
    print("=" * 80)
    
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found: {DB_PATH}")
        return False
    
    print(f"✓ Database found: {DB_PATH}")
    print(f"  Size: {os.path.getsize(DB_PATH) / 1024 / 1024:.2f} MB")
    
    # Connect to database using Sqlite3DbManager
    schema_path = str(project_root / "schemas" / "fts_schema.sql")
    db_manager = Sqlite3DbManager(
        db_path=DB_PATH,
        schema_file=schema_path
    )
    Session = db_manager.session_factory
    
    with Session() as session:
        # Count documents
        doc_count = session.query(func.count(Document.id)).scalar()
        print(f"\n📄 Documents: {doc_count}")
        
        if doc_count > 0:
            # Show document details
            documents = session.query(Document).all()
            for doc in documents[:10]:  # Show first 10
                print(f"   • {doc.file_name} (ID: {doc.id}, Type: {doc.file_type})")
            if doc_count > 10:
                print(f"   ... and {doc_count - 10} more")
        
        # Count chunks
        chunk_count = session.query(func.count(Chunk.id)).scalar()
        print(f"\n📑 Chunks: {chunk_count}")
        
        if chunk_count > 0:
            # Average chunk size
            avg_chunk_size = session.query(
                func.avg(func.length(Chunk.chunk_text))
            ).scalar()
            print(f"   Average size: {avg_chunk_size:.0f} characters")
            
            # Chunk distribution by page
            chunks_per_page = session.query(
                Chunk.page,
                func.count(Chunk.id)
            ).group_by(Chunk.page).order_by(Chunk.page).all()
            
            if len(chunks_per_page) > 0:
                total_pages = len(chunks_per_page)
                print(f"   Spread across: {total_pages} page(s)")
    
    return True


def check_faiss_status():
    """Check and display FAISS index statistics."""
    print("\n" + "=" * 80)
    print("FAISS INDEX STATUS")
    print("=" * 80)
    
    if not os.path.exists(FAISS_INDEX_PATH):
        print(f"❌ FAISS index not found: {FAISS_INDEX_PATH}")
        return False
    
    print(f"✓ FAISS index found: {FAISS_INDEX_PATH}")
    print(f"  Size: {os.path.getsize(FAISS_INDEX_PATH) / 1024 / 1024:.2f} MB")
    
    try:
        # Load index
        faiss_index = FaissIndex(
            index_file=FAISS_INDEX_PATH,
            dimension=DEFAULT_EMBEDDING_DIM,
            faiss_lock=None
        )
        
        if faiss_index._load():
            index_size = faiss_index.get_index_size()
            print(f"\n🔢 Vectors indexed: {index_size:,}")
            print(f"   Dimension: {DEFAULT_EMBEDDING_DIM}")
            
            # Calculate approximate memory usage
            memory_per_vector = DEFAULT_EMBEDDING_DIM * 4  # 4 bytes per float32
            total_memory_mb = (index_size * memory_per_vector) / 1024 / 1024
            print(f"   Approximate memory: {total_memory_mb:.2f} MB")
            
            return True
        else:
            print("❌ Failed to load FAISS index")
            return False
            
    except Exception as e:
        print(f"❌ Error loading FAISS index: {e}")
        return False


def check_consistency():
    """Check consistency between database and FAISS index."""
    print("\n" + "=" * 80)
    print("CONSISTENCY CHECK")
    print("=" * 80)
    
    if not os.path.exists(DB_PATH) or not os.path.exists(FAISS_INDEX_PATH):
        print("❌ Cannot check consistency - missing database or index")
        return
    
    # Get chunk count from database using Sqlite3DbManager
    schema_path = str(project_root / "schemas" / "fts_schema.sql")
    db_manager = Sqlite3DbManager(
        db_path=DB_PATH,
        schema_file=schema_path
    )
    Session = db_manager.session_factory
    
    with Session() as session:
        chunk_count = session.query(func.count(Chunk.id)).scalar()
    
    # Get vector count from FAISS
    faiss_index = FaissIndex(
        index_file=FAISS_INDEX_PATH,
        dimension=DEFAULT_EMBEDDING_DIM,
        faiss_lock=None
    )
    
    if faiss_index._load():
        vector_count = faiss_index.get_index_size()
        
        print(f"Chunks in database: {chunk_count:,}")
        print(f"Vectors in FAISS:   {vector_count:,}")
        
        if chunk_count == vector_count:
            print("\n✓ Database and FAISS index are consistent!")
        else:
            print(f"\n⚠ Inconsistency detected!")
            print(f"  Difference: {abs(chunk_count - vector_count):,} chunks")
            if chunk_count > vector_count:
                print(f"  → {chunk_count - vector_count} chunks missing embeddings")
            else:
                print(f"  → {vector_count - chunk_count} extra vectors in FAISS")


def main():
    """Main function to check all statuses."""
    print("\n" + "=" * 80)
    print("RAG PIPELINE - INDEX STATUS CHECK")
    print("=" * 80)
    print()
    
    # Check if we're in the right directory
    if not os.path.exists(DB_PATH) and not os.path.exists(FAISS_INDEX_PATH):
        print("⚠ No index files found in current directory.")
        print("  Make sure you're running this from the project root.")
        print(f"  Current directory: {os.getcwd()}")
        return
    
    # Check database
    db_ok = check_database_status()
    
    # Check FAISS
    faiss_ok = check_faiss_status()
    
    # Check consistency
    if db_ok and faiss_ok:
        check_consistency()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if db_ok and faiss_ok:
        print("✓ System is ready for queries!")
    elif db_ok and not faiss_ok:
        print("⚠ Database exists but FAISS index missing or corrupted")
        print("  Run Stage 2 of indexing to generate embeddings")
    elif not db_ok and faiss_ok:
        print("⚠ FAISS index exists but database missing or corrupted")
        print("  Run Stage 1 of indexing to process documents")
    else:
        print("❌ No index found. Run main_index.py to index documents.")
    
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()

