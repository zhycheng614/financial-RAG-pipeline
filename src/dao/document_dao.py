from sqlalchemy.orm import sessionmaker
from typing import List, Optional
from dao.base_dao import BaseDao
from models.document import Document


class DocumentDao(BaseDao[Document]):
    def __init__(self, session_factory: sessionmaker):
        super().__init__(session_factory, Document)
    
    def get_document_by_path(self, document_path: str) -> Optional[Document]:
        """Get a document by its file path."""
        with self.session_scope() as session:
            return session.query(Document).filter(Document.document_path == document_path).first()
    
    def get_documents_by_paths(self, document_paths: List[str]) -> List[Document]:
        """Get multiple documents by their file paths."""
        if not document_paths:
            return []
        
        with self.session_scope() as session:
            return session.query(Document).filter(Document.document_path.in_(document_paths)).all()
    
    def delete_document_by_path(self, document_path: str) -> bool:
        """Delete a document by its file path. Returns True if deleted, False otherwise."""
        with self.session_scope() as session:
            document = session.query(Document).filter(Document.document_path == document_path).first()
            if document:
                session.delete(document)
                return True
            return False
    
    def document_exists(self, document_path: str) -> bool:
        """Check if a document exists by its file path."""
        with self.session_scope() as session:
            return session.query(Document).filter(Document.document_path == document_path).first() is not None

    def get_documents_by_file_names(self, file_names: List[str]) -> List[Document]:
        """Get documents matching any of the given file names.
        
        Useful as a fallback when exact document_path matching fails
        (e.g., when the indexing path differs from the query-time path).
        
        Args:
            file_names: List of file names to search for (e.g., ["AAPL.pdf", "MSFT.pdf"])
            
        Returns:
            List of matching Document objects
        """
        if not file_names:
            return []
        
        with self.session_scope() as session:
            return session.query(Document).filter(Document.file_name.in_(file_names)).all()

