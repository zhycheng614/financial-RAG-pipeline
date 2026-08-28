from dataclasses import dataclass
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from typing import Dict, Any
from models.base import Base


@dataclass
class Document(Base):
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    document_path = Column(String, nullable=False)
    file_name = Column(String)
    file_size = Column(Integer)
    file_author = Column(String)
    file_type = Column(String)
    
    # Define the relationship with cascade delete
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

    def __eq__(self, other):
        return self.id == other.id
   
    def __hash__(self):
        return hash((self.id, self.document_path))
        
    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of the document column and values."""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
               
    def get_readable_file_size(self, file_size: int) -> str:
        """Convert file size in bytes to human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if file_size < 1024:
                return f"{file_size:.1f}{unit}"
            file_size /= 1024
        return f"{file_size:.1f}TB"