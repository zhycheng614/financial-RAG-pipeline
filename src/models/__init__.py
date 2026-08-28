# Import all models so they are registered with SQLAlchemy
from models.base import Base
from models.document import Document
from models.chunk import Chunk

__all__ = ['Base', 'Document', 'Chunk']

