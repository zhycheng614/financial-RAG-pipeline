"""
Test suite for Document model and DocumentDao
"""
import unittest
import os
import tempfile
from beans.sqlite3_db_manager import Sqlite3DbManager
from dao.document_dao import DocumentDao
from models.document import Document


class TestDocumentModel(unittest.TestCase):
    """Test Document model functionality"""
    
    def test_document_creation(self):
        """Test creating a Document instance"""
        doc = Document(
            document_path="/path/to/doc.pdf",
            file_name="doc.pdf",
            file_size=1024,
            file_author="John Doe",
            file_type="pdf"
        )
        self.assertEqual(doc.document_path, "/path/to/doc.pdf")
        self.assertEqual(doc.file_name, "doc.pdf")
        self.assertEqual(doc.file_size, 1024)
        self.assertEqual(doc.file_author, "John Doe")
        self.assertEqual(doc.file_type, "pdf")
    
    def test_document_to_dict(self):
        """Test converting document to dictionary"""
        doc = Document(
            id=1,
            document_path="/path/to/doc.pdf",
            file_name="doc.pdf",
            file_size=1024,
            file_author="John Doe",
            file_type="pdf"
        )
        doc_dict = doc.to_dict()
        self.assertIsInstance(doc_dict, dict)
        self.assertEqual(doc_dict['document_path'], "/path/to/doc.pdf")
    
    def test_document_equality(self):
        """Test document equality comparison"""
        doc1 = Document(id=1, document_path="/path/to/doc.pdf")
        doc2 = Document(id=1, document_path="/path/to/doc.pdf")
        doc3 = Document(id=2, document_path="/path/to/doc.pdf")
        
        self.assertEqual(doc1, doc2)
        self.assertNotEqual(doc1, doc3)
    
    def test_document_hash(self):
        """Test document hash function"""
        doc1 = Document(id=1, document_path="/path/to/doc.pdf")
        doc2 = Document(id=1, document_path="/path/to/doc.pdf")
        doc3 = Document(id=2, document_path="/path/to/doc.pdf")
        
        self.assertEqual(hash(doc1), hash(doc2))
        self.assertNotEqual(hash(doc1), hash(doc3))
    
    def test_get_readable_file_size(self):
        """Test converting file size to human-readable format"""
        doc = Document()
        
        self.assertEqual(doc.get_readable_file_size(500), "500.0B")
        self.assertEqual(doc.get_readable_file_size(1024), "1.0KB")
        self.assertEqual(doc.get_readable_file_size(1024 * 1024), "1.0MB")
        self.assertEqual(doc.get_readable_file_size(1024 * 1024 * 1024), "1.0GB")


class TestDocumentDao(unittest.TestCase):
    """Test DocumentDao CRUD operations"""
    
    def setUp(self):
        """Create a temporary database for testing"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.schema_file = "schemas/fts_schema.sql"
        self.db_manager = Sqlite3DbManager(self.db_path, self.schema_file)
        self.document_dao = DocumentDao(self.db_manager.session_factory)
    
    def tearDown(self):
        """Clean up temporary database"""
        if self.db_manager.conn:
            self.db_manager.conn.close()
        # Dispose the SQLAlchemy engine too: its pooled connections keep the
        # file open, and Windows refuses os.remove() on an open file.
        if getattr(self.db_manager, 'engine', None) is not None:
            self.db_manager.engine.dispose()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_add_document(self):
        """Test adding a single document"""
        doc = Document(
            document_path="/path/to/doc1.pdf",
            file_name="doc1.pdf",
            file_size=1024,
            file_author="Author 1",
            file_type="pdf"
        )
        doc_id = self.document_dao.add(doc)
        self.assertIsNotNone(doc_id)
        self.assertGreater(doc_id, 0)
        
        # Verify document was added
        retrieved = self.document_dao.get_by_id(doc_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.document_path, "/path/to/doc1.pdf")
    
    def test_add_document_return_obj(self):
        """Test adding document and returning the object"""
        doc = Document(
            document_path="/path/to/doc2.pdf",
            file_name="doc2.pdf",
            file_size=2048,
            file_author="Author 2",
            file_type="pdf"
        )
        returned_doc = self.document_dao.add_return_obj(doc)
        self.assertIsNotNone(returned_doc)
        self.assertIsNotNone(returned_doc.id)
        self.assertEqual(returned_doc.document_path, "/path/to/doc2.pdf")
    
    def test_add_multiple_documents(self):
        """Test adding multiple documents"""
        docs = [
            Document(
                document_path=f"/path/to/doc{i}.pdf",
                file_name=f"doc{i}.pdf",
                file_size=1024 * i,
                file_author=f"Author {i}",
                file_type="pdf"
            )
            for i in range(1, 4)
        ]
        doc_ids = self.document_dao.add_all(docs)
        
        self.assertEqual(len(doc_ids), 3)
        self.assertTrue(all(doc_id > 0 for doc_id in doc_ids))
        
        # Verify all documents were added
        retrieved_docs = self.document_dao.get_by_ids(doc_ids)
        self.assertEqual(len(retrieved_docs), 3)
    
    def test_get_document_by_id(self):
        """Test retrieving a document by ID"""
        doc = Document(
            document_path="/path/to/doc3.pdf",
            file_name="doc3.pdf"
        )
        doc_id = self.document_dao.add(doc)
        
        retrieved = self.document_dao.get_by_id(doc_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, doc_id)
        self.assertEqual(retrieved.document_path, "/path/to/doc3.pdf")
    
    def test_get_document_by_path(self):
        """Test retrieving a document by its file path"""
        doc = Document(
            document_path="/unique/path/to/doc.pdf",
            file_name="doc.pdf"
        )
        self.document_dao.add(doc)
        
        retrieved = self.document_dao.get_document_by_path("/unique/path/to/doc.pdf")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.document_path, "/unique/path/to/doc.pdf")
    
    def test_get_documents_by_paths(self):
        """Test retrieving multiple documents by their paths"""
        paths = [f"/path/to/doc{i}.pdf" for i in range(1, 4)]
        docs = [Document(document_path=path, file_name=os.path.basename(path)) for path in paths]
        self.document_dao.add_all(docs)
        
        retrieved = self.document_dao.get_documents_by_paths(paths)
        self.assertEqual(len(retrieved), 3)
        retrieved_paths = [doc.document_path for doc in retrieved]
        for path in paths:
            self.assertIn(path, retrieved_paths)
    
    def test_get_all_documents(self):
        """Test retrieving all documents"""
        docs = [
            Document(document_path=f"/path/to/doc{i}.pdf", file_name=f"doc{i}.pdf")
            for i in range(1, 6)
        ]
        self.document_dao.add_all(docs)
        
        all_docs = self.document_dao.get_all()
        self.assertGreaterEqual(len(all_docs), 5)
    
    def test_update_document_by_id(self):
        """Test updating a document by ID"""
        doc = Document(
            document_path="/path/to/doc4.pdf",
            file_name="doc4.pdf",
            file_size=1024
        )
        doc_id = self.document_dao.add(doc)
        
        # Update the document
        updated = self.document_dao.update_by_id(doc_id, {
            'file_size': 2048,
            'file_author': 'Updated Author'
        })
        self.assertTrue(updated)
        
        # Verify the update
        retrieved = self.document_dao.get_by_id(doc_id)
        self.assertEqual(retrieved.file_size, 2048)
        self.assertEqual(retrieved.file_author, 'Updated Author')
    
    def test_update_documents_by_ids(self):
        """Test updating multiple documents by IDs"""
        docs = [
            Document(document_path=f"/path/to/doc{i}.pdf", file_name=f"doc{i}.pdf")
            for i in range(1, 4)
        ]
        doc_ids = self.document_dao.add_all(docs)
        
        # Update all documents
        self.document_dao.update_by_ids(doc_ids, {'file_author': 'Batch Author'})
        
        # Verify the updates
        retrieved_docs = self.document_dao.get_by_ids(doc_ids)
        for doc in retrieved_docs:
            self.assertEqual(doc.file_author, 'Batch Author')
    
    def test_delete_document_by_id(self):
        """Test deleting a document by ID"""
        doc = Document(
            document_path="/path/to/doc5.pdf",
            file_name="doc5.pdf"
        )
        doc_id = self.document_dao.add(doc)
        
        # Delete the document
        self.document_dao.delete_by_ids([doc_id])
        
        # Verify deletion
        retrieved = self.document_dao.get_by_id(doc_id)
        self.assertIsNone(retrieved)
    
    def test_delete_document_by_path(self):
        """Test deleting a document by its file path"""
        doc = Document(
            document_path="/path/to/delete_me.pdf",
            file_name="delete_me.pdf"
        )
        self.document_dao.add(doc)
        
        # Delete by path
        deleted = self.document_dao.delete_document_by_path("/path/to/delete_me.pdf")
        self.assertTrue(deleted)
        
        # Verify deletion
        retrieved = self.document_dao.get_document_by_path("/path/to/delete_me.pdf")
        self.assertIsNone(retrieved)
    
    def test_delete_multiple_documents(self):
        """Test deleting multiple documents"""
        docs = [
            Document(document_path=f"/path/to/doc{i}.pdf", file_name=f"doc{i}.pdf")
            for i in range(1, 4)
        ]
        doc_ids = self.document_dao.add_all(docs)
        
        # Delete all documents
        self.document_dao.delete_by_ids(doc_ids)
        
        # Verify deletions
        retrieved_docs = self.document_dao.get_by_ids(doc_ids)
        self.assertEqual(len(retrieved_docs), 0)
    
    def test_document_exists(self):
        """Test checking if a document exists"""
        doc = Document(
            document_path="/path/to/exists.pdf",
            file_name="exists.pdf"
        )
        self.document_dao.add(doc)
        
        self.assertTrue(self.document_dao.document_exists("/path/to/exists.pdf"))
        self.assertFalse(self.document_dao.document_exists("/path/to/not_exists.pdf"))
    
    def test_get_nonexistent_document(self):
        """Test retrieving a non-existent document returns None"""
        retrieved = self.document_dao.get_by_id(99999)
        self.assertIsNone(retrieved)
    
    def test_update_nonexistent_document(self):
        """Test updating a non-existent document returns False"""
        updated = self.document_dao.update_by_id(99999, {'file_size': 1024})
        self.assertFalse(updated)
    
    def test_delete_nonexistent_document(self):
        """Test deleting a non-existent document by path returns False"""
        deleted = self.document_dao.delete_document_by_path("/nonexistent/path.pdf")
        self.assertFalse(deleted)


if __name__ == '__main__':
    unittest.main()

