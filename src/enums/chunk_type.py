from enum import Enum

class ChunkType(Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    OTHER = "other"