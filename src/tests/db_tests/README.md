# Database Tests

This directory contains comprehensive tests for the database models, DAOs, and connection management.

## Test Files

### test_db_connection.py
Tests for database connection and schema setup:
- Database file creation
- Table creation (documents, chunks)
- FTS virtual table creation (chunks_fts)
- Trigger creation for automatic FTS synchronization
- WAL mode enablement
- SQLAlchemy session factory and engine creation

### test_document.py
Tests for Document model and DocumentDao:
- **Document Model Tests:**
  - Document creation and initialization
  - to_dict() method
  - Equality and hash functions
  - File size formatting

- **DocumentDao CRUD Tests:**
  - Adding single and multiple documents
  - Retrieving documents by ID
  - Retrieving documents by file path
  - Retrieving multiple documents by paths
  - Getting all documents
  - Updating documents by ID
  - Bulk updating multiple documents
  - Deleting documents by ID
  - Deleting documents by file path
  - Checking document existence
  - Edge cases (non-existent documents)

### test_chunk.py
Tests for Chunk model and ChunkDao:
- **Chunk Model Tests:**
  - Chunk creation with simplified fields (id, document_id, chunk_text, file, page)
  - to_dict() method
  - Text preview generation (short and long text)

- **ChunkDao CRUD Tests:**
  - Adding single and multiple chunks
  - Retrieving chunks by document ID
  - Retrieving chunks by multiple document IDs
  - Getting chunk IDs by document IDs
  - Getting chunk IDs by file path
  - Updating chunks
  - Deleting chunks
  - Deleting chunks by document ID
  - Bulk deleting chunks by multiple document IDs
  - Cascade delete (chunks deleted when document is deleted)

- **FTS Search Tests:**
  - Basic full-text search
  - Search with BM25 scores
  - Search with subset of chunk IDs
  - Search with empty subset
  - Search with no results
  - Keyword/phrase search
  - Keyword search with exclusions
  - Keyword search with file filters
  - Searching file field

- **FTS Trigger Tests:**
  - Automatic FTS update on insert
  - Automatic FTS update on update
  - Automatic FTS delete on delete

## Running the Tests

### Run all database tests:
```bash
python -m unittest discover tests/db_tests
```

### Run a specific test file:
```bash
python -m unittest tests.db_tests.test_db_connection
python -m unittest tests.db_tests.test_document
python -m unittest tests.db_tests.test_chunk
```

### Run a specific test class:
```bash
python -m unittest tests.db_tests.test_document.TestDocumentDao
python -m unittest tests.db_tests.test_chunk.TestChunkDao
```

### Run a specific test method:
```bash
python -m unittest tests.db_tests.test_chunk.TestChunkDao.test_fts_search_basic
```

### Run with verbose output:
```bash
python -m unittest discover tests/db_tests -v
```

## Test Coverage

The tests cover:
- ✅ All CRUD operations (Create, Read, Update, Delete)
- ✅ Database connection and schema setup
- ✅ SQLAlchemy ORM functionality
- ✅ FTS5 full-text search with BM25 ranking
- ✅ FTS triggers for automatic synchronization
- ✅ Relationship cascades (document -> chunks)
- ✅ Batch operations
- ✅ Edge cases and error conditions
- ✅ Query filtering and subsetting
- ✅ Metadata field indexing

## Notes

- All tests use temporary databases that are cleaned up after each test
- Tests are isolated and can run in any order
- The schema file `schemas/fts_schema.sql` must exist in the project root
- Tests verify both the ORM operations and the underlying FTS functionality

