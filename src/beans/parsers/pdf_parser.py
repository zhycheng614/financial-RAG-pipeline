from pypdfium2 import PdfDocument
from typing import List
from typing_extensions import override
import logging

from beans.parsers.base_parser import BaseParser
from data_classes.pdf_page import PDFPage



logger = logging.getLogger(__name__)


class PDFParser(BaseParser):
    """
    Pure PDF parser for extracting content from PDF files.
    
    This parser is responsible only for extracting text and tables from PDFs.
    It has no knowledge of chunking, embedding, or processing strategies.
    """
    def __init__(self, pdf_path: str):
        """ Initialize the PDF parser with a file path. """
        super().__init__(pdf_path)
        self.pdf_doc = PdfDocument(pdf_path)
        self.max_page_number = len(self.pdf_doc)
        self.table_coverage_threshold = 0.5
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures PDF is closed."""
        self.close()
        return False
    
    def close(self):
        """Explicitly close the PDF document to release file handles."""
        if hasattr(self, 'pdf_doc') and self.pdf_doc is not None:
            try:
                self.pdf_doc.close()
                logger.debug(f"Closed PDF document: {self.file_path}")
            except Exception as e:
                logger.warning(f"Error closing PDF document {self.file_path}: {str(e)}")
            finally:
                self.pdf_doc = None
    
    def __del__(self):
        """Destructor to ensure PDF is closed when object is garbage collected."""
        self.close()
    
    def _get_number_of_pages(self, pdf_path: str) -> int:
        """ Get the number of pages in the PDF file. """
        return self.max_page_number

    def _validate_page_number(self, page_number: int) -> None:
        """ Validate that the page number is valid (>= 1). """
        if page_number < 1:
            logger.error(f"Invalid page number: {page_number}")
            raise ValueError("Page number must be 1 or greater")
        if page_number > self.max_page_number:
            logger.error(f"Page number exceeds maximum page number: {page_number}")
            raise ValueError(f"Page number exceeds maximum page number: {page_number}")
        
    def _process_page(self, page_index: int) -> PDFPage:
        """Process a page and extract text and tables.
        
        Args:
            page_index: Zero-based page index
            
        Returns:
            PDFPage object with extracted content
        """
        page_number = page_index + 1
        logger.debug(f"Processing page {page_number}:")

        page = self.pdf_doc[page_index]
        text = page.get_textpage().get_text_range().strip()
        text = text.replace('\x01', ' ').replace('\r', '')

        return PDFPage(
            page_number=page_number,
            text=text,
            tables=[],
        )

    @override
    def parse(self) -> List[PDFPage]:
        """Parse the PDF file and return a list of PDFPage objects.
        
        This is the main parsing method that extracts text and tables from the PDF.
        
        Returns:
            List of PDFPage objects, each containing page_number, text, and tables
        """
        logger.debug(f"Parsing {self.file_path}")
        pages = [self._process_page(page_index) for page_index in range(self.max_page_number)]
        return pages
    
    def extract_all_text(self) -> List[str]:
        """
        Extract text from all pages of the PDF file.
        Note this is not suitable for very large files as it will load all the text into memory.
        """
        results = []
        try:
            pages = self.parse()
            results = [page.text for page in pages]
            return results
        except Exception as e:
            import traceback
            logger.error(f"Error extracting all text from PDF '{self.file_path}': {str(e)}\n{traceback.format_exc()}")
            return results


if __name__ == "__main__":
    # Configure logging to see debug messages
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # PDF path
    pdf_path = r"data/test_pdfs/example.pdf"
    
    print(f"Testing PDFParser with: {pdf_path}")
    print("=" * 80)
    
    # Use context manager to ensure proper cleanup
    with PDFParser(pdf_path) as parser:
        # Method 1: Extract all text as strings
        print("\n" + "="*80)
        print("METHOD 1: extract_all_text() - Returns List[str]")
        print("="*80)
        text_pages = parser.extract_all_text()
        
        print(f"\nExtracted {len(text_pages)} pages\n")
        
        for i, page_text in enumerate(text_pages, 1):
            print(f"\n{'='*80}")
            print(f"PAGE {i}")
            print(f"{'='*80}")
            print(page_text)
            print(f"\n(Page {i} - {len(page_text)} characters)")
        
        # Method 2: Parse to PDFPage objects and chunk them
        print("\n\n" + "="*80)
        print("METHOD 2: parse() + Chunker - Returns PDFPage objects & ParsedChunks")
        print("="*80)
        
        # Parse PDF into PDFPage objects
        pdf_pages = parser.parse()
        print(f"\nParsed {len(pdf_pages)} PDFPage objects")
        
        # Chunk the PDFPage objects using universal Chunker
        from beans.chunker import Chunker
        from constants import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
        
        chunker = Chunker(
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP
        )
        
        parsed_chunks = chunker.chunk_iterable(
            parsed_elements=pdf_pages,
            file_path=pdf_path,
            document_id=None  # Optional, can be set when saving to DB
        )
        
        print(f"\nChunking Results:")
        print(f"  - Text chunks: {len(parsed_chunks.text)}")
        print(f"  - Table chunks: {len(parsed_chunks.tables)}")
        print(f"  - Full table chunks: {len(parsed_chunks.full_tables)}")
        
        # Display text chunks with page numbers
        print(f"\n{'='*80}")
        print("TEXT CHUNKS (with page numbers preserved):")
        print("="*80)
        for i, chunk in enumerate(parsed_chunks.text, 1):
            print(f"\nChunk {i} (Page {chunk.page}):")
            print(f"  File: {chunk.file}")
            print(f"  Text preview: {chunk.chunk_text[:200]}...")
            print(f"  Full length: {len(chunk.chunk_text)} characters")
    
    print(f"\n{'='*80}")
    print("Parsing and chunking complete!")
    print("="*80)