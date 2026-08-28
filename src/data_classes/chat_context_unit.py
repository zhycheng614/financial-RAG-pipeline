from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timezone
from models.document import Document
from data_classes.chunk_info import ChunkInfo

@dataclass
class ChatContextUnit:
    chunk_content: str
    chunk_source: str
    document: Optional[Document] = None
    current_time: datetime = datetime.now(timezone.utc)
    # fields below are not used in __str__
    chunk_id: int = None
    document_id: int = None
    rerank_score: float = None
    rrf_score: float = None
    rel_pos: int = None
    
    def __str__(self) -> str:
        """Format the chat context unit metadata into a string."""
        file_metadata = (
            f"source: {self.chunk_source}"
            # f"(size: {self.document.get_readable_file_size()}), " +
            # f"created at {self._format_datetime(self.document.creation_time)}, " +
            # f"modified at {self._format_datetime(self.document.last_modified_time)} " +
            # f"(The current time is {self._format_datetime(self.current_time)})"
        )
        return f"{file_metadata}. Content:\n{self.chunk_content}"
        
    @staticmethod
    def _format_datetime(time: datetime):
        # Return format as YYYY-MM-DD Hour-Minute 
        return time.strftime("%Y-%m-%d %H:%M")
    
    
    def get_debuggable_str(self) -> str:
        rrf_score_str = f"{self.rrf_score:.3f}" if self.rrf_score is not None else "N/A"
        rerank_score_str = f"{self.rerank_score:.3f}" if self.rerank_score is not None else "N/A"
        return f"Chunk-{self.chunk_id}, File name: {self.chunk_source}, (RRF score: {rrf_score_str}, Rerank score: {rerank_score_str})"
    
    # This class needs to be hashabble
    # TODO: this hash may not be unique anymore, please be careful with using it.
    def __hash__(self):
        # Use chunk_id, document_id, and chunk_content for hashing
        return hash(str(self))
    
    def __eq__(self, other):
        if not isinstance(other, ChunkInfo):
            return False
        
        return str(self) == str(other)