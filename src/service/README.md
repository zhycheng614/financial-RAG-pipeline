# Document Processor Service

## Overview

The `DocumentProcessor` provides a complete pipeline for processing documents: **parse → chunk → save to database**.

The architecture is designed to be **scalable and extensible**, making it easy to add support for new file types.

## Architecture

### Pipeline Stages

```
Document File
    ↓
1. Parse (file type specific)
    ↓
2. Chunk (preserves page numbers)
    ↓
3. Save to Database (Document + Chunks)
```

### Key Components

- **DocumentProcessor**: Main orchestrator
- **FileType**: Enum for file type detection
- **Parsers**: File-specific parsers (e.g., PDFParser)
- **Chunker**: Splits content into manageable chunks
- **DAOs**: Database access objects (DocumentDao, ChunkDao)

## Usage

### Basic Usage

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from service.document_processor import DocumentProcessor
from dao.document_dao import DocumentDao
from dao.chunk_dao import ChunkDao
from beans.chunker import Chunker
from constants import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

# Setup database connection
engine = create_engine('sqlite:///my_database.db')
Session = sessionmaker(bind=engine)

# Initialize dependencies (Dependency Injection)
document_dao = DocumentDao(Session)
chunk_dao = ChunkDao(Session)
chunker = Chunker(
    text_chunk_size=DEFAULT_CHUNK_SIZE,
    text_chunk_overlap=DEFAULT_CHUNK_OVERLAP
)

# Initialize processor with injected dependencies
processor = DocumentProcessor(
    document_dao=document_dao,
    chunk_dao=chunk_dao,
    chunker=chunker,
    parser_registry=None  # Uses default registry with PDF support
)

# Process a document
document = processor.process_document(
    file_path="path/to/document.pdf",
    overwrite=False  # Skip if already processed
)

if document:
    print(f"Processed document ID: {document.id}")
```

### Processing with Overwrite

```python
# Reprocess an existing document
document = processor.process_document(
    file_path="path/to/document.pdf",
    overwrite=True  # Delete and reprocess
)
```

### Accessing Processed Chunks

```python
from dao.chunk_dao import ChunkDao

chunk_dao = ChunkDao(Session)

# Get all chunks for a document
chunks = chunk_dao.get_chunks_by_document_id(document.id)

# Each chunk preserves the original page number
for chunk in chunks:
    print(f"Page {chunk.page}: {chunk.chunk_text[:100]}...")
```

## Extending with New File Types

The system uses a **Parser Registry** pattern for easy extension without modifying core code.

### Step 1: Add File Type to Enum

```python
# enums/file_type.py
class FileType(Enum):
    PDF = "pdf"
    DOCX = "docx"  # New file type
    HTML = "html"  # Another new file type
    OTHER = "other"
```

### Step 2: Create a New Parser

```python
# beans/parsers/docx_parser.py
from beans.parsers.base_parser import BaseParser
from data_classes.pdf_page import PDFPage  # Or create DocxPage
from typing import List

class DOCXParser(BaseParser):
    """Parser for DOCX files."""
    
    def __init__(self, file_path: str):
        super().__init__(file_path)
        # Initialize DOCX-specific resources
    
    def __enter__(self):
        """Context manager support (optional but recommended)."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup resources."""
        return False
    
    def parse(self) -> List[PDFPage]:
        """Parse DOCX and return page-like objects."""
        # Implementation here
        pass
```

### Step 3: Register the Parser

**Option A: Using Default Registry (modify at initialization)**

```python
from enums.file_type import FileType
from beans.parsers.pdf_parser import PDFParser
from beans.parsers.docx_parser import DOCXParser

# Create custom parser registry
parser_registry = {
    FileType.PDF: PDFParser,
    FileType.DOCX: DOCXParser,  # Add new parser
}

# Initialize processor with custom registry
processor = DocumentProcessor(
    document_dao=document_dao,
    chunk_dao=chunk_dao,
    chunker=chunker,
    parser_registry=parser_registry
)
```

**Option B: Register Dynamically**

```python
from enums.file_type import FileType
from beans.parsers.docx_parser import DOCXParser

# Initialize processor with defaults
processor = DocumentProcessor(
    document_dao=document_dao,
    chunk_dao=chunk_dao,
    chunker=chunker
)

# Register new parser dynamically
processor.register_parser(FileType.DOCX, DOCXParser)

# Now DOCX files are supported!
document = processor.process_document("document.docx")
```

### Step 4: Add Chunking Method (if needed)

If your new file type needs custom chunking:

```python
# beans/chunker.py

def chunk_docx_pages(self, docx_pages: List, file_path: str, document_id: Optional[int] = None):
    """Chunk DOCX pages into ParsedChunks."""
    # Implementation similar to chunk_pdf_pages
    pass
```

Then update `_chunk_document` in `DocumentProcessor`:

```python
def _chunk_document(self, parsed_pages: List, file_path: str, file_type: FileType):
    if file_type == FileType.PDF:
        return self.chunker.chunk_pdf_pages(...)
    elif file_type == FileType.DOCX:
        return self.chunker.chunk_docx_pages(...)
    else:
        raise ValueError(f"Unsupported file type for chunking: {file_type.value}")
```

## Current Support

### Supported File Types

- ✅ **PDF** - Full support with page number preservation

### Coming Soon

- 📄 DOCX
- 🌐 HTML
- 📊 XLSX (Excel)
- 📝 TXT

## API Reference

### DocumentProcessor

#### `__init__(document_dao, chunk_dao, chunker, parser_registry=None)`

Initialize the processor with dependency injection.

**Parameters:**
- `document_dao` (DocumentDao): DAO for document operations
- `chunk_dao` (ChunkDao): DAO for chunk operations
- `chunker` (Chunker): Chunker instance for splitting content
- `parser_registry` (Dict[FileType, Type[BaseParser]], optional): Custom parser registry

#### `register_parser(file_type, parser_class)`

Dynamically register a new parser for a file type.

**Parameters:**
- `file_type` (FileType): The file type enum value
- `parser_class` (Type[BaseParser]): Parser class inheriting from BaseParser

#### `process_document(file_path, overwrite=False)`

Process a single document through the complete pipeline.

**Parameters:**
- `file_path` (str): Path to the document file
- `overwrite` (bool): If True, reprocess even if exists

**Returns:**
- `Document`: Document object if successful, None otherwise

**Raises:**
- `FileNotFoundError`: If file doesn't exist
- `ValueError`: If file type is not supported

## Testing

Run the built-in test:

```bash
python service/document_processor.py
```

This will:
1. Create a test database (`test_document_processor.db`)
2. Process a sample PDF
3. Display statistics and sample chunks
4. Show page number distribution

## Database Schema

### Documents Table

| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary key |
| document_path | String | Full path to file |
| file_name | String | File name |
| file_size | Integer | Size in bytes |
| file_type | String | File type (pdf, docx, etc.) |
| file_author | String | Author (optional) |

### Chunks Table

| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary key |
| document_id | Integer | Foreign key to documents |
| chunk_text | Text | Chunk content |
| file | String | Source file path |
| page | Integer | Original page number |

## Best Practices

### 1. Dependency Injection

Always inject dependencies rather than creating them internally:

```python
# Good: Dependency Injection
document_dao = DocumentDao(Session)
chunk_dao = ChunkDao(Session)
chunker = Chunker(chunk_size=2500, chunk_overlap=1250)

processor = DocumentProcessor(
    document_dao=document_dao,
    chunk_dao=chunk_dao,
    chunker=chunker
)

# This makes testing easier:
# - Can inject mock DAOs for testing
# - Can inject different chunkers with different settings
# - More flexible and maintainable
```

### 2. Use Context Managers in Parsers

Parsers should be used with context managers to ensure proper cleanup:

```python
class MyParser(BaseParser):
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup resources
        return False
```

The DocumentProcessor automatically detects and uses context managers.

### 3. Check for Existing Documents

Before processing, check if a document exists:

```python
from dao.document_dao import DocumentDao

document_dao = DocumentDao(Session)
existing = document_dao.get_document_by_path(file_path)

if existing:
    print(f"Already processed: {existing.id}")
else:
    document = processor.process_document(file_path)
```

### 4. Handle Errors Gracefully

The processor includes automatic rollback on errors:

```python
try:
    document = processor.process_document(file_path)
    if document:
        print("Success!")
    else:
        print("Processing failed - check logs")
except FileNotFoundError:
    print("File not found")
except ValueError as e:
    print(f"Unsupported file type: {e}")
```

### 5. Batch Processing

For multiple documents:

```python
import os
from pathlib import Path

pdf_directory = Path("data/test_pdfs")

for pdf_file in pdf_directory.glob("*.pdf"):
    try:
        document = processor.process_document(
            file_path=str(pdf_file),
            overwrite=False  # Skip already processed
        )
        if document:
            print(f"✓ {pdf_file.name}")
        else:
            print(f"✗ {pdf_file.name}")
    except Exception as e:
        print(f"✗ {pdf_file.name}: {str(e)}")
```

## Logging

The processor uses Python's standard logging:

```python
import logging

# Configure logging level
logging.basicConfig(level=logging.INFO)

# Or for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

Log levels:
- **INFO**: Processing progress and statistics
- **DEBUG**: Detailed parsing information
- **ERROR**: Failures and exceptions

## Performance Considerations

### Chunk Size

- **Larger chunks** (3000-5000): Better context, fewer DB records
- **Smaller chunks** (1000-2000): More granular retrieval, more records

### Overlap

- **More overlap** (50%): Better context continuity, more redundancy
- **Less overlap** (10-20%): Less redundancy, smaller database

### Recommended Settings

```python
# For general documents
chunker = Chunker(
    text_chunk_size=2500,
    text_chunk_overlap=1250  # 50% overlap
)

# For large documents
chunker = Chunker(
    text_chunk_size=4000,
    text_chunk_overlap=800  # 20% overlap
)

# For precise retrieval
chunker = Chunker(
    text_chunk_size=1500,
    text_chunk_overlap=750  # 50% overlap
)

# Then inject into processor
processor = DocumentProcessor(
    document_dao=document_dao,
    chunk_dao=chunk_dao,
    chunker=chunker
)
```

## Troubleshooting

### "File type not supported"

Add the file extension to `FileType` enum and implement corresponding parser.

### "Failed to create document record"

Check database connection and ensure tables are created:

```python
from models.base import Base
Base.metadata.create_all(engine)
```

### Memory Issues with Large PDFs

The current implementation loads pages sequentially, but for very large PDFs, consider:
- Processing in batches
- Streaming chunks to database
- Increasing available memory

### Page Numbers Not Preserved

Ensure your parser returns objects with `page_number` attribute and that chunker uses `split_paginated_text()`.

## Future Enhancements

- [ ] Async processing for large batches
- [ ] Progress callbacks
- [ ] Document metadata extraction (author, creation date, etc.)
- [ ] OCR support for scanned PDFs
- [ ] Table extraction and special handling
- [ ] Image extraction from documents
- [ ] Document versioning
- [ ] Incremental updates (process only changed pages)

