from typing import List, Optional
from dataclasses import dataclass
from constants import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
import re

PAGE_BREAK_PATTERN = "[PAGE{PAGE_NUMBER}]"

@dataclass
class TextChunkOnPage:
    """A chunk of text from a specific element/page.
    
    Note: element_index is a 1-based index into the source elements array,
    NOT the actual page number. Use chunker to map to actual page numbers.
    """
    text: str
    index_on_page: int  # Index of this chunk within the element (0-based)
    element_index: int  # Which element/section this chunk came from (1-based)


class TextSplitter:
    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        length_function = len,
        is_separator_regex: bool = False,
        separators: Optional[List[str]] = None
    ):
        """Initialize a text splitter with customizable parameters.
           With the default configurations, this means that each chunk will be 2500 characters long and will overlap by 250 characters.

        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Number of characters to overlap between chunks
            length_function: Function to measure text length
            is_separator_regex: Whether separators are regex patterns
            separators: List of separator strings to split on
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.is_separator_regex = is_separator_regex
        self.separators = separators or [
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "",
        ]

        self._create_text_splitter(self.chunk_size, self.chunk_overlap)

    def _create_text_splitter(self, chunk_size: int, chunk_overlap: int):
        """Create the underlying text splitter with specified parameters."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self.length_function,
            is_separator_regex=self.is_separator_regex,
            separators=self.separators,
        )

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks based on configured parameters.

        Args:
            text: The text to split

        Returns:
            A list of text chunks
        """
        return self.text_splitter.split_text(text)

    def split_with_page_info(self, text: str, num_elements: Optional[int] = None) -> List[TextChunkOnPage]:
        """Split text into chunks while preserving element/section information.
        
        Assigns each chunk to an element based on where the chunk starts in the merged text.
        If a chunk spans multiple elements, it's assigned to the element where it begins.

        Args:
            text: The merged text with element break markers
            num_elements: Total number of source elements in the document

        Returns:
            List of TextChunkOnPage objects with element_index indicating source element (1-based)
        """
        # Step 1: Find all element marker positions and create position-to-element mapping
        element_marker_positions = []
        for elem_idx in range(1, num_elements + 1):
            marker = PAGE_BREAK_PATTERN.format(PAGE_NUMBER=elem_idx)
            pos = text.find(marker)
            if pos != -1:
                element_marker_positions.append((pos, elem_idx))
        
        # Sort by position
        element_marker_positions.sort(key=lambda x: x[0])
        
        # Step 2: Create a mapping from clean text positions to original positions with markers
        # Build a list mapping clean_pos -> original_pos
        clean_text = self.remove_page_break_pattern(text)
        
        # Create position mapping: for each position in clean text, track original position
        clean_to_original = []
        clean_idx = 0
        original_idx = 0
        
        while original_idx < len(text):
            # Check if we're at a marker
            is_marker = False
            for marker_pos, marker_elem in element_marker_positions:
                marker = PAGE_BREAK_PATTERN.format(PAGE_NUMBER=marker_elem)
                if original_idx == marker_pos and text[original_idx:original_idx + len(marker)] == marker:
                    # Skip the marker in original text
                    original_idx += len(marker)
                    is_marker = True
                    break
            
            if not is_marker and original_idx < len(text):
                # This character is in the clean text
                clean_to_original.append(original_idx)
                clean_idx += 1
                original_idx += 1
        
        # Step 3: Split the clean text (without markers)
        chunks = self.split_text(clean_text)
        
        # Step 4: Assign each chunk to an element based on where it starts
        output_chunks = []
        current_clean_pos = 0
        element_chunk_counts = [0] * (num_elements + 1)  # Track chunks per element
        
        for chunk_text in chunks:
            # Find this chunk in the clean text
            chunk_start_clean = clean_text.find(chunk_text, current_clean_pos)
            
            if chunk_start_clean == -1:
                # Fallback
                element_idx = 1
                index_on_page = element_chunk_counts[element_idx]
            else:
                # Map clean position to original position
                if chunk_start_clean < len(clean_to_original):
                    chunk_start_original = clean_to_original[chunk_start_clean]
                else:
                    chunk_start_original = len(text) - 1
                
                # Determine which element this chunk belongs to
                element_idx = 1  # Default to first element
                
                # Find the last element marker that appears before this chunk
                for marker_pos, marker_elem in element_marker_positions:
                    if marker_pos < chunk_start_original:
                        element_idx = marker_elem + 1  # Chunk is after this marker
                    else:
                        break
                
                # Ensure element_idx doesn't exceed num_elements
                if element_idx > num_elements:
                    element_idx = num_elements
                
                index_on_page = element_chunk_counts[element_idx]
                
                # Update position for next search
                current_clean_pos = chunk_start_clean + len(chunk_text)
            
            text_chunk = TextChunkOnPage(
                text=chunk_text,  # Already clean, no markers
                index_on_page=index_on_page,
                element_index=element_idx  # 1-based index into source elements array
            )
            output_chunks.append(text_chunk)
            element_chunk_counts[element_idx] += 1
        
        return output_chunks

    def remove_page_break_pattern(self, text: str) -> str:
        """Remove the page break pattern from the text with any number that substitutes for the page number.
        """
        # Match patterns like [PAGE1], [PAGE2], etc.
        pattern = r'\[PAGE\d+\]'
        return re.sub(pattern, "", text)

    def merge_paginated_text(self, paginated_text: List[str]) -> str:
        """Merge text from multiple pages with page break markers.

        Args:
            paginated_text: List of text strings, one per page

        Returns:
            Combined text with page break markers
        """
        merged_text = ""
        for i, text in enumerate(paginated_text):
            page_number = i + 1
            merged_text += text
            merged_text += "\n\n" + PAGE_BREAK_PATTERN.format(PAGE_NUMBER=page_number) + "\n\n"
        return merged_text

    def split_paginated_text(self, paginated_text: List[str]) -> List[TextChunkOnPage]:
        """Split text from multiple elements, preserving element information.

        Args:
            paginated_text: List of text strings, one per element/page

        Returns:
            List of TextChunkOnPage objects with element_index (1-based) indicating 
            which element from paginated_text the chunk came from
        """
        num_elements = len(paginated_text)
        merged_text = self.merge_paginated_text(paginated_text)
        return self.split_with_page_info(merged_text, num_elements)
