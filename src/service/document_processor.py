import logging
import os
from typing import Optional, List, Dict, Type
from pathlib import Path
from sqlalchemy.orm import sessionmaker
import numpy as np

from enums.file_type import FileType
from beans.parsers.base_parser import BaseParser
from beans.chunker import Chunker
from beans.embedder import Embedder
from beans.faiss_manager import FaissIndex
from dao.document_dao import DocumentDao
from dao.chunk_dao import ChunkDao
from models.document import Document
from models.chunk import Chunk
from data_classes.parsed_chunks import ParsedChunks
from constants import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Process documents through the full pipeline: parse → chunk → save to database.
    
    This processor is designed to be scalable and supports multiple file types.
    Uses a universal Chunker that works with any parsed document type.
    """
    
    def __init__(
        self,
        document_dao: DocumentDao,
        chunk_dao: ChunkDao,
        embedder: Embedder,
        faiss_index: Optional[FaissIndex] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        parser_registry: Optional[Dict[FileType, Type[BaseParser]]] = None
    ):
        """Initialize the document processor.
        
        Args:
            document_dao: DAO for document database operations
            chunk_dao: DAO for chunk database operations
            embedder: Embedder instance for generating embeddings
            faiss_index: FaissIndex instance for vector storage (optional, only needed for embedding operations)
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between consecutive chunks
            parser_registry: Dictionary mapping FileType to Parser classes
                            If None, uses default registry
        """
        self.document_dao = document_dao
        self.chunk_dao = chunk_dao
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedder = embedder
        self.faiss_index = faiss_index
        
        # Create universal chunker (works for all file types)
        self.chunker = Chunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Use provided parser registry or create default one
        self.parser_registry = parser_registry or self._create_default_parser_registry()
    
    def _create_default_parser_registry(self) -> Dict[FileType, Type[BaseParser]]:
        """Create default parser registry with built-in parsers.
        
        Returns:
            Dictionary mapping FileType to Parser class
        """
        from beans.parsers.pdf_parser import PDFParser
        
        return {
            FileType.PDF: PDFParser,
            # Future parsers can be added here:
            # FileType.DOCX: DOCXParser,
            # FileType.HTML: HTMLParser,
        }
    
    def register_parser(self, file_type: FileType, parser_class: Type[BaseParser]) -> None:
        """Register a new parser for a file type.
        
        This allows adding support for new file types without modifying the processor.
        
        Args:
            file_type: The file type to register
            parser_class: Parser class (must inherit from BaseParser)
        """
        if not issubclass(parser_class, BaseParser):
            raise ValueError(f"Parser class must inherit from BaseParser")
        
        self.parser_registry[file_type] = parser_class
        logger.info(f"Registered parser {parser_class.__name__} for {file_type.value}")

    def _parse_document(self, file_path: str, file_type: FileType) -> Optional[List]:
        """Parse document based on file type using registered parsers.
        
        Uses the parser registry to dynamically select and instantiate the
        appropriate parser. This approach leverages polymorphism through the
        BaseParser interface, making it easy to add new file types.
        
        Args:
            file_path: Path to the document
            file_type: Type of the file (from FileType enum)
            
        Returns:
            List of parsed pages/content, or None if parsing fails
            
        Raises:
            ValueError: If file type is not supported (no parser registered)
        """
        # Check if parser is registered for this file type
        if file_type not in self.parser_registry:
            raise ValueError(
                f"No parser registered for file type: {file_type.value}. "
                f"Available types: {[ft.value for ft in self.parser_registry.keys()]}"
            )
        
        try:
            # Get parser class from registry
            parser_class = self.parser_registry[file_type]
            
            # Instantiate and use parser (using context manager if available)
            parser = parser_class(file_path)
            
            # Check if parser supports context manager
            if hasattr(parser, '__enter__') and hasattr(parser, '__exit__'):
                with parser:
                    return parser.parse()
            else:
                # Use parser directly if no context manager
                return parser.parse()
                
        except Exception as e:
            logger.error(f"Error parsing document {file_path}: {str(e)}", exc_info=True)
            return None
    
    def _chunk_document(
        self,
        parsed_pages: List,
        file_path: str,
        file_type: FileType
    ) -> ParsedChunks:
        """Chunk parsed document content using universal chunker.
        
        Uses the universal Chunker which works with any parsed element type
        that has a 'text' field. Optional 'page_number' and 'tables' fields
        are preserved/chunked if present.
        
        Args:
            parsed_pages: Parsed content from the parser
            file_path: Original file path
            file_type: Type of the file (for logging)
            
        Returns:
            ParsedChunks object containing chunked content
        """
        try:
            # Use universal chunker to process the parsed content
            return self.chunker.chunk_iterable(
                parsed_elements=parsed_pages,
                file_path=file_path,
                document_id=None  # Will be set after document is created
            )
            
        except Exception as e:
            logger.error(f"Error chunking document {file_path}: {str(e)}", exc_info=True)
            raise
    
    def _save_to_database(
        self,
        file_path: str,
        file_type: FileType,
        parsed_chunks: ParsedChunks
    ) -> Optional[Document]:
        """Save document and chunks to database.
        
        Args:
            file_path: Path to the document file
            file_type: Type of the file
            parsed_chunks: Chunked content to save
            
        Returns:
            Document object if successful, None otherwise
        """
        try:
            # Extract file metadata
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            
            # Create document record
            document = Document(
                document_path=file_path,
                file_name=file_name,
                file_size=file_size,
                file_type=file_type.value,
                file_author=None  # Could be extracted from PDF metadata if needed
            )
            
            # Save document to get ID
            document = self.document_dao.add_return_obj(document)
            
            if not document or not document.id:
                logger.error("Failed to create document record in database")
                return None
            
            # Update chunk document_ids and prepare for bulk insert
            all_chunks: List[Chunk] = []
            
            # Add text chunks
            for chunk in parsed_chunks.text:
                chunk.document_id = document.id
                all_chunks.append(chunk)
            
            # Add table chunks
            for chunk in parsed_chunks.tables:
                chunk.document_id = document.id
                all_chunks.append(chunk)
            
            # Add full table chunks
            for chunk in parsed_chunks.full_tables:
                chunk.document_id = document.id
                all_chunks.append(chunk)
            
            # Bulk save chunks
            if all_chunks:
                chunk_ids = self.chunk_dao.add_all(all_chunks)
                logger.info(f"Saved {len(chunk_ids)} chunks to database")
            
            return document
            
        except Exception as e:
            logger.error(f"Error saving to database: {str(e)}", exc_info=True)
            # Attempt to clean up document if chunks failed
            if 'document' in locals() and document and document.id:
                try:
                    self.document_dao.delete_by_ids([document.id])
                    logger.info("Rolled back document creation due to error")
                except Exception as cleanup_error:
                    logger.error(f"Failed to clean up document: {str(cleanup_error)}")
            return None

    def process_document(
        self,
        file_path: str,
        overwrite: bool = False
    ) -> Optional[Document]:
        """
        Process a single document through the complete pipeline.
        
        Pipeline stages:
        1. Validate file exists and detect file type
        2. Check if document already exists in database
        3. Parse document using appropriate parser
        4. Chunk parsed content
        5. Save document and chunks to database
        
        Args:
            file_path: Path to the document file
            overwrite: If True, reprocess document even if it exists in DB
            
        Returns:
            Document object if successful, None otherwise
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file type is not supported
        """
        logger.info(f"Processing document: {file_path}")
        
        # Step 1: Validate file and detect type
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_path = os.path.abspath(file_path)
        file_type = FileType.from_filename(file_path)
        
        logger.info(f"Detected file type: {file_type.value}")
        
        # Step 2: Check if document already exists
        existing_document = self.document_dao.get_document_by_path(file_path)
        
        if existing_document and not overwrite:
            logger.info(f"Document already exists in database (ID: {existing_document.id}). Skipping.")
            return existing_document
        
        if existing_document and overwrite:
            logger.info(f"Document exists (ID: {existing_document.id}). Overwriting...")
            # Delete existing document (chunks will be cascade deleted)
            self.document_dao.delete_by_ids([existing_document.id])
        
        # Step 3: Parse document
        logger.info("Parsing document...")
        parsed_pages = self._parse_document(file_path, file_type)
        
        if not parsed_pages:
            logger.error(f"Failed to parse document: {file_path}")
            return None
        
        logger.info(f"Successfully parsed {len(parsed_pages)} pages")
        
        # Step 4: Chunk the parsed content
        logger.info("Chunking document...")
        parsed_chunks = self._chunk_document(parsed_pages, file_path, file_type)
        
        total_chunks = (
            len(parsed_chunks.text) + 
            len(parsed_chunks.tables) + 
            len(parsed_chunks.full_tables)
        )
        logger.info(f"Created {total_chunks} chunks ({len(parsed_chunks.text)} text, "
                   f"{len(parsed_chunks.tables)} table, {len(parsed_chunks.full_tables)} full table)")
        
        # Step 5: Save to database
        logger.info("Saving to database...")
        document = self._save_to_database(file_path, file_type, parsed_chunks)
        
        if document:
            logger.info(f"Successfully processed document (ID: {document.id}) with {total_chunks} chunks")
        else:
            logger.error(f"Failed to save document to database: {file_path}")
        
        return document
    
    def calculate_embeddings_for_chunks_and_add_to_vector_db(self, document_id: int) -> List[int]:
        """
        Calculate embeddings for all chunks of a document and add them to the FAISS vector database.
        
        This method:
        1. Fetches all chunks for the given document ID
        2. Generates embeddings for each chunk's text using the embedder
        3. Adds the embeddings to the FAISS index with chunk IDs
        
        Args:
            document_id: ID of the document whose chunks need embeddings
            
        Returns:
            List of chunk IDs that were successfully embedded and added to the index
        """
        if self.faiss_index is None:
            logger.error("FaissIndex is not initialized. Cannot calculate embeddings.")
            raise ValueError("FaissIndex must be provided to calculate embeddings.")
        
        logger.info(f"Calculating embeddings for document {document_id}")
        
        # Fetch chunks for the document
        chunks = self.chunk_dao.get_chunks_by_document_id(document_id)
        logger.debug(f"Fetched {len(chunks)} chunks for document {document_id} for embedding calculation")
        
        if len(chunks) == 0:
            logger.warning(f"No chunks found for document {document_id}. Skipping embedding calculation.")
            return []
        
        # Extract chunk IDs and texts
        chunk_ids = [chunk.id for chunk in chunks]
        chunk_texts = [chunk.chunk_text for chunk in chunks]
        
        # Generate embeddings in batch
        logger.info(f"Generating embeddings for {len(chunk_texts)} chunks...")
        embedding_vectors = self.embedder.generate_embeddings(chunk_texts)
        embedding_ids = np.array(chunk_ids, dtype=np.int64)
        
        # Add embeddings to FAISS index
        logger.info(f"Adding embeddings to FAISS index...")
        completed_chunk_ids = self.faiss_index.add_embeddings(embedding_vectors, embedding_ids)
        logger.info(f"Successfully added {len(completed_chunk_ids)} embeddings for document {document_id} to the vector index.")
        
        return completed_chunk_ids


if __name__ == "__main__":
    """Test the document processor with a sample PDF."""
    import logging
    from models.base import Base
    from sqlalchemy import create_engine
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Database setup (using SQLite for testing)
    DB_PATH = "test_document_processor.db"
    engine = create_engine(f'sqlite:///{DB_PATH}', echo=False)
    
    # Create tables
    Base.metadata.create_all(engine)
    
    # Create session factory
    Session = sessionmaker(bind=engine)
    
    print("="*80)
    print("DOCUMENT PROCESSOR TEST")
    print("="*80)
    
    # Initialize dependencies
    document_dao = DocumentDao(Session)
    chunk_dao = ChunkDao(Session)
    
    # Initialize embedder
    from data_classes.embedding_generator_config import EmbedderConfig
    embedder_config = EmbedderConfig()
    embedder = Embedder(embedder_config)
    
    # Initialize FAISS index
    from constants import DEFAULT_EMBEDDING_DIM
    faiss_index_file = "test_faiss_index.index"
    faiss_index = FaissIndex(
        index_file=faiss_index_file,
        dimension=DEFAULT_EMBEDDING_DIM,
        faiss_lock=None  # No lock needed for single-process testing
    )
    
    # Initialize processor
    processor = DocumentProcessor(
        document_dao=document_dao,
        chunk_dao=chunk_dao,
        embedder=embedder,
        faiss_index=faiss_index,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP
    )
    
    print(f"\nRegistered parsers:")
    for file_type in processor.parser_registry:
        parser_class = processor.parser_registry[file_type]
        print(f"  - {file_type.value}: {parser_class.__name__}")
    
    print(f"\nUniversal Chunker:")
    print(f"  - Chunk size: {processor.chunker.chunk_size}")
    print(f"  - Chunk overlap: {processor.chunker.chunk_overlap}")
    print(f"  - Works with any file type (text field required)")
    
    # Test PDF path
    pdf_path = r"data/test_pdfs/example.pdf"
    
    print(f"\nProcessing: {pdf_path}\n")
    
    # Process the document
    try:
        document = processor.process_document(
            file_path=pdf_path,
            overwrite=True  # Force reprocessing for testing
        )
        
        if document:
            print("\n" + "="*80)
            print("PROCESSING SUCCESSFUL!")
            print("="*80)
            print(f"\nDocument Details:")
            print(f"  ID: {document.id}")
            print(f"  Name: {document.file_name}")
            print(f"  Path: {document.document_path}")
            print(f"  Type: {document.file_type}")
            print(f"  Size: {document.get_readable_file_size(document.file_size)}")
            
            # Retrieve and display chunks
            chunk_dao = ChunkDao(Session)
            chunks = chunk_dao.get_chunks_by_document_id(document.id)
            
            print(f"\nChunks: {len(chunks)} total")
            
            # Show page distribution
            from collections import Counter
            page_distribution = Counter([chunk.page for chunk in chunks])
            
            print(f"\nChunks per page:")
            for page_num in sorted(page_distribution.keys()):
                count = page_distribution[page_num]
                print(f"  Page {page_num}: {count} chunk(s)")
            
            # Show sample chunks
            print(f"\nSample Chunks:")
            for i, chunk in enumerate(chunks[:3], 1):
                print(f"\n[Chunk {i}]")
                print(f"  Page: {chunk.page}")
                print(f"  Length: {len(chunk.chunk_text)} chars")
                preview = chunk.chunk_text[:100].replace('\n', ' ')
                print(f"  Preview: \"{preview}...\"")
        else:
            print("\n" + "="*80)
            print("PROCESSING FAILED!")
            print("="*80)
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        logger.error("Processing failed", exc_info=True)
    
    print("\n" + "="*80)
    print(f"Test database saved to: {DB_PATH}")
    print("="*80)

