import logging
from typing import List, Optional
from constants import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

logger = logging.getLogger(__name__)


class Chunker:
    """Universal chunker that can handle any parsed document type.
    
    Works with any iterable of elements that have:
    - Required: 'text' field (text content)
    - Optional: 'page_number' field (preserved in chunks if present)
    - Optional: 'tables' field (chunked if present)
    """
    
    def __init__(
        self, 
        chunk_size: int = DEFAULT_CHUNK_SIZE, 
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    ):
        """Initialize the chunker with size parameters.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        from beans.text_splitter import TextSplitter
        
        text_splitter = TextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return text_splitter.split_text(text)
    
    def chunk_table(self, parsed_table) -> List[str]:
        """Split a table into text chunks.
        
        Args:
            parsed_table: ParsedTable object
            
        Returns:
            List of table chunk strings
        """
        from beans.table_splitter import TableSplitter
        
        table_splitter = TableSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return table_splitter.split_table_into_text_chunks(parsed_table)
    
    def chunk_iterable(
        self,
        parsed_elements: List,
        file_path: str,
        document_id: Optional[int] = None
    ):
        """Chunk an iterable of parsed elements into ParsedChunks.
        
        This method merges all text first, then splits as a whole while preserving
        page numbers. This ensures chunks can span across element boundaries naturally.
        
        Page Number Preservation:
        - Elements can have any page_number values (e.g., [1, 2, 4, 5, 6])
        - Text is merged with element markers: "text1[PAGE1]text2[PAGE2]text4[PAGE3]..."
        - Chunks get element_index (1-based) indicating which element they came from
        - element_index is then mapped to actual page_number from the source element
        
        Example:
            Elements: [PDFPage(page_number=1, text="..."), 
                      PDFPage(page_number=2, text="..."),
                      PDFPage(page_number=4, text="...")]
            → Chunks with element_index=1 get page=1
            → Chunks with element_index=2 get page=2
            → Chunks with element_index=3 get page=4  (not 3!)
        
        Elements must have a 'text' field; 'page_number' and 'tables' are optional.
        
        Args:
            parsed_elements: List of parsed elements (e.g., PDFPage, DocxPage, etc.)
            file_path: Path to the source file
            document_id: Optional document ID for database association
            
        Returns:
            ParsedChunks object containing text and table chunks with metadata
        """
        from beans.text_splitter import TextSplitter
        from data_classes.parsed_chunks import ParsedChunks
        from models.chunk import Chunk
        
        text_chunks = []
        table_chunks = []
        full_table_chunks = []
        
        # Create text splitter for chunking
        text_splitter = TextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # Step 1: Extract all text from elements (in order)
        # Example: Elements have page_numbers [1, 2, 4, 5, 6] (page 3 might be missing/skipped)
        paginated_text = []
        element_page_numbers = []  # Maps element index → actual page number
        
        for element in parsed_elements:
            # Extract required field: text
            if not hasattr(element, 'text'):
                logger.error(f"Element missing 'text' field.")
                raise ValueError(f"Element missing 'text' field.")
            
            text = element.text
            page_number = getattr(element, 'page_number', None)
            
            paginated_text.append(text if text else "")
            element_page_numbers.append(page_number)
        
        # Now: paginated_text = [text_from_p1, text_from_p2, text_from_p4, text_from_p5, text_from_p6]
        #      element_page_numbers = [1, 2, 4, 5, 6]
        
        # Step 2: Merge and split text as a whole, preserving page information
        # This uses split_paginated_text which merges with page markers then splits
        text_chunks_on_pages = text_splitter.split_paginated_text(paginated_text)
        
        # Step 3: Convert TextChunkOnPage to Chunk objects
        # Note: element_index is a 1-based index into the paginated_text array
        for text_chunk_on_page in text_chunks_on_pages:
            # Map element index (1-based) to array index (0-based)
            array_index = text_chunk_on_page.element_index - 1
            
            # Get the actual page number from the corresponding element
            # Handle edge cases where index might be out of bounds
            if 0 <= array_index < len(element_page_numbers):
                actual_page_number = element_page_numbers[array_index]
            else:
                logger.warning(f"Chunk element index {array_index} out of bounds for element_page_numbers")
                actual_page_number = None
            
            chunk = Chunk(
                document_id=document_id,
                chunk_text=text_chunk_on_page.text,
                file=file_path,
                page=actual_page_number  # Actual page number from original element
            )
            text_chunks.append(chunk)
        
        # Step 4: Process tables separately (per element, not merged)
        for element in parsed_elements:
            page_number = getattr(element, 'page_number', None)
            
            if hasattr(element, 'tables'):
                tables = element.tables
                if tables:
                    for table in tables:
                        # Split large tables into smaller chunks
                        table_text_chunks = self.chunk_table(table)
                        for table_text in table_text_chunks:
                            chunk = Chunk(
                                document_id=document_id,
                                chunk_text=table_text,
                                file=file_path,
                                page=page_number
                            )
                            table_chunks.append(chunk)
                        
                        # Also store full table representation
                        full_table_text = table.to_text()
                        full_table_chunk = Chunk(
                            document_id=document_id,
                            chunk_text=full_table_text,
                            file=file_path,
                            page=page_number
                        )
                        full_table_chunks.append(full_table_chunk)
        
        logger.info(f"Created {len(text_chunks)} text chunks, {len(table_chunks)} table chunks, "
                   f"{len(full_table_chunks)} full table chunks")
        
        return ParsedChunks(
            text=text_chunks,
            tables=table_chunks,
            full_tables=full_table_chunks
        )


if __name__ == "__main__":
    """Test the chunker with a sample PDF."""
    import logging
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test PDF path
    pdf_path = r"data/test_pdfs/example.pdf"
    
    print("="*80)
    print("CHUNKER TEST: PDF Parsing + Chunking with Page Number Preservation")
    print("="*80)
    print(f"\nPDF: {pdf_path}\n")
    
    # Step 1: Parse the PDF
    print("STEP 1: Parsing PDF...")
    print("-"*80)
    
    from beans.parsers.pdf_parser import PDFParser
    
    with PDFParser(pdf_path) as parser:
        pdf_pages = parser.parse()
        print(f"✓ Successfully parsed {len(pdf_pages)} pages")
        
        # Show page info
        for page in pdf_pages:
            char_count = len(page.text)
            table_count = len(page.tables)
            print(f"  - Page {page.page_number}: {char_count} characters, {table_count} tables")
        
        # Step 2: Chunk the PDF pages using Chunker
        print(f"\n{'='*80}")
        print("STEP 2: Chunking PDF pages with universal Chunker...")
        print("-"*80)
        
        # Use universal Chunker
        chunker = Chunker(
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP
        )
        
        parsed_chunks = chunker.chunk_iterable(
            parsed_elements=pdf_pages,
            file_path=pdf_path,
            document_id=1  # Example document ID
        )
        
        print(f"✓ Chunking complete!")
        print(f"\nResults:")
        print(f"  - Text chunks: {len(parsed_chunks.text)}")
        print(f"  - Table chunks: {len(parsed_chunks.tables)}")
        print(f"  - Full table chunks: {len(parsed_chunks.full_tables)}")
        
        # Step 3: Display chunks with page numbers
        print(f"\n{'='*80}")
        print("STEP 3: Text Chunks with Page Numbers")
        print("="*80)
        
        for i, chunk in enumerate(parsed_chunks.text[:5], 1):  # Show first 5
            print(f"\n[Chunk {i}]")
            print(f"  Page Number: {chunk.page}")
            print(f"  Document ID: {chunk.document_id}")
            print(f"  File: {chunk.file}")
            print(f"  Chunk Length: {len(chunk.chunk_text)} characters")
            print(f"  Text Preview:")
            # Show first 150 characters
            preview = chunk.chunk_text[:150].replace('\n', ' ')
            print(f"    \"{preview}...\"")
        
        # Verify page numbers are preserved
        print(f"\n{'='*80}")
        print("VERIFICATION: Page Number Distribution")
        print("="*80)
        
        from collections import Counter
        page_distribution = Counter([chunk.page for chunk in parsed_chunks.text])
        
        print(f"\nChunks per page:")
        for page_num in sorted(page_distribution.keys()):
            count = page_distribution[page_num]
            print(f"  Page {page_num}: {count} chunk(s)")
        
        print(f"\n{'='*80}")
        print("✓ Test Complete! Page numbers successfully preserved in chunks.")
        print("="*80)

