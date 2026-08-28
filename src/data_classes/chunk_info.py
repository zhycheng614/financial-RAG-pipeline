from dataclasses import dataclass

@dataclass
class ChunkInfo:
    chunk_id: int
    document_id: int = None
    chunk_content: str = None
    
    # This class needs to be hashabble
    def __hash__(self):
        # Use chunk_id, document_id, and chunk_content for hashing
        return hash((self.chunk_id, self.document_id, self.chunk_content))
    
    def __eq__(self, other):
        if not isinstance(other, ChunkInfo):
            return False

        return (self.chunk_id, self.document_id, self.chunk_content) == (other.chunk_id, other.document_id, other.chunk_content)