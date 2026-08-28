from dataclasses import dataclass
from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship
from typing import Dict, Any
from models.base import Base

@dataclass
class Chunk(Base):
    __tablename__ = 'chunks'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey('documents.id', ondelete='CASCADE'), nullable=False)
    chunk_text = Column(Text, nullable=False)
    file = Column(String, nullable=False)
    page = Column(Integer)
    
    document = relationship("Document", back_populates="chunks")

    
    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of the chunk column and values."""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
    
    def get_text_preview(self, max_length: int = 100) -> str:
        """Get a preview of the chunk text."""
        if len(self.chunk_text) <= max_length:
            return self.chunk_text
        return self.chunk_text[:max_length] + "..."