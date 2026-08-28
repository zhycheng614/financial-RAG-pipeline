from dataclasses import dataclass
from typing import List
from dataclasses import field
from models.chunk import Chunk

@dataclass
class ParsedChunks():
    text: List[Chunk] = field(default_factory=list)
    tables: List[Chunk] = field(default_factory=list)
    full_tables: List[Chunk] = field(default_factory=list)