

from dataclasses import dataclass
from typing import List
from data_classes.parsed_table import ParsedTable

@dataclass
class PDFPage():
    page_number: int
    text: str
    tables: List[ParsedTable]