from typing import List, Optional
from data_classes.parsed_table import ParsedTable

class TableSplitter:
    def __init__(
        self,
        chunk_size: int = 100,
        chunk_overlap: int = 10,
        length_function = len,
        is_separator_regex: bool = False,
        separators: Optional[List[str]] = None,
    ):
        """Initialize a text splitter with customizable parameters.

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

        from beans.text_splitter import TextSplitter
        self.text_splitter = TextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
            is_separator_regex=self.is_separator_regex,
            separators=separators,
        )

    def split_table_into_text_chunks(self, table: ParsedTable) -> List[str]:
        """Split table into chunks based on configured parameters.

        Args:
            table: The table to split

        Returns:
            A list of table chunks
        """
        table_str = table.describe_table_row_by_row()
        table_str_chunks = self.text_splitter.split_text(table_str)
        table_header_description = table.describe_headers().strip()
        # add the header description to every chunk except the first one
        for i in range(1, len(table_str_chunks)):
            table_str_chunks[i] = table_header_description + "\n" + table_str_chunks[i].strip()
        return table_str_chunks