# RAG Pipeline - Main Scripts

This folder contains the main entry points for the RAG pipeline.

## Scripts

### `main_index.py` - Document Indexing Pipeline

A concurrent document indexing pipeline that processes multiple documents in parallel.

#### Features

1. **Parallel Document Processing**: Uses `ProcessPoolExecutor` to parse and chunk documents in parallel
2. **Async Embedding Generation**: Uses `asyncio` to generate embeddings concurrently via OpenAI API
3. **Thread-Safe FAISS Writes**: Uses multiprocessing locks to prevent race conditions when writing to FAISS
4. **SQLite Database**: Stores document metadata and chunks (no lock needed - SQLite handles concurrency)

#### Usage

```bash
# Show help
python main/main_index.py -h

# Basic: Index a folder
python main/main_index.py /path/to/documents

# Interactive mode (prompts for path)
python main/main_index.py

# Overwrite existing documents
python main/main_index.py /path/to/documents --overwrite

# Adjust concurrency
python main/main_index.py /path/to/documents --workers 4 --async-tasks 10

# Enable verbose logging
python main/main_index.py /path/to/documents --verbose
```

#### Command-Line Options

```
positional arguments:
  folder                Path to folder containing documents to index

optional arguments:
  -h, --help            Show help message and exit
  -o, --overwrite       Overwrite existing documents in the database
  -w N, --workers N     Number of worker processes for Stage 1 (default: CPU count)
  -a N, --async-tasks N Number of concurrent async tasks for Stage 2 (default: 5)
  -v, --verbose         Enable verbose logging (DEBUG level)
```

#### Pipeline Stages

**Stage 1: Document Processing (Parallel)**
- Uses multiple CPU cores to process documents in parallel
- Each process:
  - Validates and detects file type
  - Parses document content
  - Chunks the content
  - Saves document and chunks to SQLite database

**Stage 2: Embedding Generation (Async)**
- Uses asyncio for concurrent API calls to OpenAI
- Each async task:
  - Fetches chunks for a document
  - Generates embeddings via OpenAI API (async)
  - Adds embeddings to FAISS index (with lock)
- Default: 5 concurrent async tasks (configurable)

#### Configuration

Key parameters can be adjusted via command-line options or in the script:

**Via Command-Line:**
```bash
# Adjust workers and async tasks
python main/main_index.py /path/to/docs --workers 8 --async-tasks 10
```

**In the script (main_index.py):**
```python
# Database and index paths
DB_PATH = "rag_pipeline.db"
FAISS_INDEX_PATH = "faiss_index.index"

# Default concurrency settings (can be overridden by --workers and --async-tasks)
# num_workers: default = CPU count
# max_concurrent_tasks: default = 5
```

#### Supported File Types

Currently supported file types are defined in `enums/file_type.py`:
- PDF (`.pdf`)

Additional file types can be added by:
1. Adding the enum to `FileType`
2. Creating a parser that inherits from `BaseParser`
3. Registering the parser in `DocumentProcessor`

#### Output

The pipeline creates two files in the project root:
- `rag_pipeline.db` - SQLite database with documents and chunks
- `faiss_index.index` - FAISS vector index for similarity search

#### Example Output

```
================================================================================
RAG PIPELINE - CONCURRENT DOCUMENT INDEXING
================================================================================
Indexing folder: /path/to/documents
Searching for supported files...
Found 10 supported file(s)
  1. document1.pdf
  2. document2.pdf
  ...

================================================================================
STAGE 1: Processing documents (parse + chunk)
================================================================================
Using 8 worker processes
✓ Processed: document1.pdf (ID: 1)
✓ Processed: document2.pdf (ID: 2)
...

Stage 1 complete: 10/10 documents processed

================================================================================
STAGE 2: Calculating embeddings and building vector index
================================================================================
✓ [Async] Completed embeddings for document1.pdf (45 chunks)
✓ [Async] Completed embeddings for document2.pdf (32 chunks)
...

Stage 2 complete: 10 successful, 0 failed

================================================================================
INDEXING COMPLETE
================================================================================
Documents found:      10
Documents processed:  10
Embeddings generated: 10
Embeddings failed:    0

Database: rag_pipeline.db
FAISS Index: faiss_index.index
================================================================================
```

### `main_query.py` - Query Processing Pipeline

A concurrent query processing pipeline that handles multiple queries with RAG-based answer generation.

#### Features

1. **Multiple Input Sources**: 
   - CLI (single query)
   - CSV file (multiple queries from a column)
   - HuggingFace dataset (multiple queries from a column)
2. **Async Query Processing**: Uses `asyncio` to process multiple queries concurrently
3. **RAG Answer Generation**: Retrieves relevant chunks and generates answers using LLM
4. **Flexible Output**: Terminal display or CSV file export
5. **Configurable Pipeline**: Enable/disable rewriting, retrieval, and reranking

#### Usage Examples

```bash
# Show help
python main/main_query.py -h

# Single query from CLI (terminal output)
python main/main_query.py --query "What is machine learning?"

# Multiple queries from CSV file
python main/main_query.py --csv queries.csv --column question

# Queries from CSV with custom ID column
python main/main_query.py --csv queries.csv --column question --id-column query_id

# Queries from HuggingFace dataset
python main/main_query.py --hf-dataset squad --column question --split validation

# Output results to CSV
python main/main_query.py --csv queries.csv --column question --output-csv results.csv

# Adjust concurrency (5 concurrent queries)
python main/main_query.py --csv queries.csv --column question --concurrent 5

# Disable query rewriting
python main/main_query.py --query "What is AI?" --no-rewrite

# Disable reranking
python main/main_query.py --query "What is AI?" --no-rerank

# Custom model and base URL
python main/main_query.py --query "What is AI?" --model llama-3.2-3b-instruct --base-url http://localhost:8000/v1
```

#### Command-Line Options

**Input Source (one required):**
```
--query TEXT              Single query from command line
--csv PATH                Path to CSV file containing queries
--hf-dataset DATASET_ID   HuggingFace dataset ID
```

**Input Configuration:**
```
--column NAME             Column name containing queries (required for CSV/HF)
--id-column NAME          Column name for query IDs (optional, defaults to row index)
--split NAME              Dataset split for HuggingFace (default: train)
```

**Output Options:**
```
--output-csv PATH         Output results to CSV file (default: terminal)
```

**Processing Configuration:**
```
--concurrent N            Number of concurrent query tasks (default: 3)
--model NAME              Model name for answer generation (default: llama-3.2-3b-instruct)
--base-url URL            Base URL for inference service (default: http://localhost:8000/v1)
```

**Pipeline Control:**
```
--no-rewrite              Disable query rewriting
--no-rerank               Disable reranking
--no-retrieve             Disable retrieval (for testing)
```

**Logging:**
```
-v, --verbose             Enable verbose logging (DEBUG level)
```

#### Pipeline Stages

For each query, the pipeline executes the following stages:

**Stage 1: Query Rewriting (optional)**
- Clarifies vague queries
- Extracts keywords for full-text search
- Uses LLM to improve query quality

**Stage 2: Retrieval**
- Hybrid search (semantic + keyword-based)
- Reciprocal Rank Fusion (RRF) to combine results
- Filters by semantic similarity threshold

**Stage 3: Reranking (optional)**
- Uses Jina reranker to score relevance
- Applies cutoff thresholds (cumulative and cliff)
- Keeps only the most relevant chunks

**Stage 4: Answer Generation**
- Formats retrieved chunks as context
- Sends to LLM with RAG system prompt
- Generates final answer based on context

All stages execute asynchronously for multiple queries to maximize throughput.

#### Input File Formats

**CSV File Example:**
```csv
id,question,category
1,What is machine learning?,ML
2,How does neural network work?,DL
3,What is reinforcement learning?,RL
```

Usage:
```bash
python main/main_query.py --csv queries.csv --column question --id-column id
```

**HuggingFace Dataset Example:**
```bash
# Using SQuAD dataset
python main/main_query.py --hf-dataset squad --column question --split validation

# Using a custom dataset
python main/main_query.py --hf-dataset username/dataset-name --column query_text --split test
```

#### Output Formats

**Terminal Output (default):**
```
================================================================================
QUERY RESULTS
================================================================================

[Query 1] ID: 1
Question: What is machine learning?
Retrieved Chunks: 5
Chunk IDs: [123, 456, 789, 101, 202]
Document IDs: [1, 2, 3]

Answer:
Machine learning is a subset of artificial intelligence that enables systems
to learn and improve from experience without being explicitly programmed...
--------------------------------------------------------------------------------
...
```

**CSV Output (--output-csv results.csv):**
```csv
query_id,query,chunk_ids,document_ids,final_answer
1,What is machine learning?,"123,456,789,101,202","1,2,3","Machine learning is..."
2,How does neural network work?,"124,457,790","1,2","A neural network is..."
```

#### Configuration

Key parameters can be adjusted in the script or via command-line:

**Via Command-Line:**
```bash
# Adjust concurrency and model
python main/main_query.py --csv queries.csv --column question \
  --concurrent 10 --model llama-3.2-3b-instruct
```

**In constants.py:**
```python
# Retriever configuration
DEFAULT_SEMANTIC_FILTERING_THRESHOLD = 2.0
DEFAULT_FINAL_CHUNKS_TO_ASSEMBLE_LIMIT = 10
DEFAULT_FTS_RETRIEVAL_LIMIT = 20
DEFAULT_SEMANTIC_RETRIEVAL_LIMIT = 30
DEFAULT_RRF_K = 60

# Reranking configuration
DEFAULT_MAX_ITEM_TO_RERANK = 30
DEFAULT_RERANK_KEEP_THRESHOLD = 0.45
DEFAULT_RERANK_CLIFF_CUTOFF_SCORE_DIFFERENCE = 0.15
```

#### System Prompt Customization

The RAG answer generation prompt can be customized in `prompts.py`:

```python
RAG_ANSWER_GENERATION_SYSTEM_PROMPT = r"""
You are a helpful assistant that answers questions based on provided context...
"""

RAG_ANSWER_GENERATION_USER_PROMPT = """\
Context from retrieved documents:
{context}

User Question: {query}
...
"""
```

#### Performance Tips

1. **Concurrency**: Increase `--concurrent` for faster processing (watch for API rate limits)
2. **Reranking**: Disable with `--no-rerank` if speed is critical (slight quality drop)
3. **Rewriting**: Disable with `--no-rewrite` for simple, clear queries
4. **Batch Processing**: Use CSV/HF input for processing many queries efficiently

#### Error Handling

- Failed queries are logged but don't stop the pipeline
- Errors are captured in the output CSV (if using --output-csv)
- All errors are logged with full stack traces for debugging
- Query results include an error field for failed queries

#### Example Workflow

1. **Index documents** (one-time setup):
   ```bash
   python main/main_index.py /path/to/documents
   ```

2. **Prepare queries** (CSV file):
   ```csv
   id,question
   1,What are the key findings?
   2,What methodology was used?
   3,What are the limitations?
   ```

3. **Process queries**:
   ```bash
   python main/main_query.py --csv queries.csv --column question --output-csv answers.csv
   ```

4. **Review results** (answers.csv):
   ```csv
   query_id,query,chunk_ids,document_ids,final_answer
   1,"What are the key findings?","45,67,89","1,2","The key findings are..."
   ...
   ```

## Notes

### Concurrency Model

1. **ProcessPoolExecutor for CPU-bound work**: Document parsing and chunking are CPU-intensive, so we use multiple processes
2. **Asyncio for I/O-bound work**: API calls to OpenAI are I/O-bound, so we use async/await
3. **Sequential stages**: Stage 2 only starts after Stage 1 completes (as specified)

### Thread Safety

- **FAISS writes**: Protected by multiprocessing lock (critical for preventing corruption)
- **SQLite writes**: No lock needed (SQLite handles concurrent writes via internal locking)

### Performance Tips

1. **Adjust worker count**: For very large files, reduce `num_workers` to avoid memory issues
2. **Adjust concurrent tasks**: For faster embedding generation, increase `max_concurrent_tasks` (watch for rate limits)
3. **Batch size**: Embedding batch size is controlled by `EMBEDDING_MAX_BATCH_SIZE` in `constants.py`

### Error Handling

- Failed documents in Stage 1 are logged but don't stop the pipeline
- Failed embeddings in Stage 2 are logged but don't stop other documents
- All errors are logged with full stack traces for debugging

