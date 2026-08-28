"""
Test suite for Chunk model and ChunkDao
"""
import unittest
import os
import tempfile
from beans.sqlite3_db_manager import Sqlite3DbManager
from dao.document_dao import DocumentDao
from dao.chunk_dao import ChunkDao
from models.document import Document
from models.chunk import Chunk


class TestChunkModel(unittest.TestCase):
    """Test Chunk model functionality"""
    
    def test_chunk_creation(self):
        """Test creating a Chunk instance"""
        chunk = Chunk(
            document_id=1,
            chunk_text="This is a test chunk",
            file="test.pdf",
            page=1
        )
        self.assertEqual(chunk.document_id, 1)
        self.assertEqual(chunk.chunk_text, "This is a test chunk")
        self.assertEqual(chunk.file, "test.pdf")
        self.assertEqual(chunk.page, 1)
    
    def test_chunk_to_dict(self):
        """Test converting chunk to dictionary"""
        chunk = Chunk(
            id=1,
            document_id=1,
            chunk_text="Test text",
            file="test.pdf"
        )
        chunk_dict = chunk.to_dict()
        self.assertIsInstance(chunk_dict, dict)
        self.assertEqual(chunk_dict['chunk_text'], "Test text")
        self.assertEqual(chunk_dict['file'], "test.pdf")
    
    def test_get_text_preview_short(self):
        """Test text preview for short text"""
        chunk = Chunk(
            document_id=1,
            chunk_text="Short text",
            file="test.pdf"
        )
        preview = chunk.get_text_preview(100)
        self.assertEqual(preview, "Short text")
    
    def test_get_text_preview_long(self):
        """Test text preview for long text"""
        long_text = "A" * 200
        chunk = Chunk(
            document_id=1,
            chunk_text=long_text,
            file="test.pdf"
        )
        preview = chunk.get_text_preview(100)
        self.assertEqual(len(preview), 103)  # 100 chars + "..."
        self.assertTrue(preview.endswith("..."))


class TestChunkDao(unittest.TestCase):
    """Test ChunkDao CRUD and search operations"""
    
    def setUp(self):
        """Create a temporary database for testing"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.schema_file = "schemas/fts_schema.sql"
        self.db_manager = Sqlite3DbManager(self.db_path, self.schema_file)
        self.document_dao = DocumentDao(self.db_manager.session_factory)
        self.chunk_dao = ChunkDao(self.db_manager.session_factory)
        
        # Create test documents
        self.doc1_id = self.document_dao.add(Document(
            document_path="/path/to/doc1.pdf",
            file_name="doc1.pdf"
        ))
        self.doc2_id = self.document_dao.add(Document(
            document_path="/path/to/doc2.pdf",
            file_name="doc2.pdf"
        ))
    
    def tearDown(self):
        """Clean up temporary database"""
        if self.db_manager.conn:
            self.db_manager.conn.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_add_chunk(self):
        """Test adding a single chunk"""
        chunk = Chunk(
            document_id=self.doc1_id,
            chunk_text="This is a test chunk with important information",
            file="doc1.pdf",
            page=1
        )
        chunk_id = self.chunk_dao.add(chunk)
        self.assertIsNotNone(chunk_id)
        self.assertGreater(chunk_id, 0)
        
        # Verify chunk was added
        retrieved = self.chunk_dao.get_by_id(chunk_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.chunk_text, "This is a test chunk with important information")
    
    def test_add_multiple_chunks(self):
        """Test adding multiple chunks"""
        chunks = [
            Chunk(
                document_id=self.doc1_id,
                chunk_text=f"Chunk {i} with some content",
                file="doc1.pdf",
                page=i
            )
            for i in range(1, 6)
        ]
        chunk_ids = self.chunk_dao.add_all(chunks)
        
        self.assertEqual(len(chunk_ids), 5)
        self.assertTrue(all(chunk_id > 0 for chunk_id in chunk_ids))
        
        # Verify all chunks were added
        retrieved_chunks = self.chunk_dao.get_by_ids(chunk_ids)
        self.assertEqual(len(retrieved_chunks), 5)
    
    def test_get_chunks_by_document_id(self):
        """Test retrieving all chunks for a specific document"""
        chunks = [
            Chunk(
                document_id=self.doc1_id,
                chunk_text=f"Doc1 Chunk {i}",
                file="doc1.pdf",
                page=i
            )
            for i in range(1, 4)
        ]
        self.chunk_dao.add_all(chunks)
        
        # Add chunks for doc2
        chunk2 = Chunk(
            document_id=self.doc2_id,
            chunk_text="Doc2 Chunk",
            file="doc2.pdf",
            page=1
        )
        self.chunk_dao.add(chunk2)
        
        # Get chunks for doc1
        doc1_chunks = self.chunk_dao.get_chunks_by_document_id(self.doc1_id)
        self.assertEqual(len(doc1_chunks), 3)
        for chunk in doc1_chunks:
            self.assertEqual(chunk.document_id, self.doc1_id)
    
    def test_get_chunks_by_document_ids(self):
        """Test retrieving chunks for multiple documents"""
        # Add chunks for doc1
        for i in range(1, 3):
            self.chunk_dao.add(Chunk(
                document_id=self.doc1_id,
                chunk_text=f"Doc1 Chunk {i}",
                file="doc1.pdf"
            ))
        
        # Add chunks for doc2
        for i in range(1, 3):
            self.chunk_dao.add(Chunk(
                document_id=self.doc2_id,
                chunk_text=f"Doc2 Chunk {i}",
                file="doc2.pdf"
            ))
        
        chunks = self.chunk_dao.get_chunks_by_document_ids([self.doc1_id, self.doc2_id])
        self.assertEqual(len(chunks), 4)
    
    def test_get_chunk_ids_by_document_ids(self):
        """Test retrieving only chunk IDs for specific documents"""
        chunk_ids = []
        for i in range(1, 4):
            chunk_id = self.chunk_dao.add(Chunk(
                document_id=self.doc1_id,
                chunk_text=f"Chunk {i}",
                file="doc1.pdf"
            ))
            chunk_ids.append(chunk_id)
        
        retrieved_ids = self.chunk_dao.get_chunk_ids_by_document_ids([self.doc1_id])
        self.assertEqual(len(retrieved_ids), 3)
        for chunk_id in chunk_ids:
            self.assertIn(chunk_id, retrieved_ids)
    
    def test_get_chunk_ids_by_file_path(self):
        """Test retrieving chunk IDs by file path"""
        chunk_ids = []
        for i in range(1, 3):
            chunk_id = self.chunk_dao.add(Chunk(
                document_id=self.doc1_id,
                chunk_text=f"Chunk {i}",
                file="doc1.pdf"
            ))
            chunk_ids.append(chunk_id)
        
        retrieved_ids = self.chunk_dao.get_chunk_ids_by_file_path("/path/to/doc1.pdf")
        self.assertEqual(len(retrieved_ids), 2)
        for chunk_id in chunk_ids:
            self.assertIn(chunk_id, retrieved_ids)
    
    def test_update_chunk(self):
        """Test updating a chunk"""
        chunk = Chunk(
            document_id=self.doc1_id,
            chunk_text="Original text",
            file="doc1.pdf"
        )
        chunk_id = self.chunk_dao.add(chunk)
        
        # Update the chunk
        updated = self.chunk_dao.update_by_id(chunk_id, {
            'chunk_text': 'Updated text',
            'page': 5
        })
        self.assertTrue(updated)
        
        # Verify the update
        retrieved = self.chunk_dao.get_by_id(chunk_id)
        self.assertEqual(retrieved.chunk_text, 'Updated text')
        self.assertEqual(retrieved.page, 5)
    
    def test_delete_chunk(self):
        """Test deleting a single chunk"""
        chunk = Chunk(
            document_id=self.doc1_id,
            chunk_text="Delete me",
            file="doc1.pdf"
        )
        chunk_id = self.chunk_dao.add(chunk)
        
        # Delete the chunk
        self.chunk_dao.delete_by_ids([chunk_id])
        
        # Verify deletion
        retrieved = self.chunk_dao.get_by_id(chunk_id)
        self.assertIsNone(retrieved)
    
    def test_delete_chunks_by_document_id(self):
        """Test deleting all chunks for a specific document"""
        # Add chunks for doc1
        for i in range(1, 4):
            self.chunk_dao.add(Chunk(
                document_id=self.doc1_id,
                chunk_text=f"Chunk {i}",
                file="doc1.pdf"
            ))
        
        # Delete all chunks for doc1
        deleted_count = self.chunk_dao.delete_chunks_by_document_id(self.doc1_id)
        self.assertEqual(deleted_count, 3)
        
        # Verify deletion
        doc1_chunks = self.chunk_dao.get_chunks_by_document_id(self.doc1_id)
        self.assertEqual(len(doc1_chunks), 0)
    
    def test_delete_chunks_by_document_ids(self):
        """Test bulk deleting chunks for multiple documents"""
        # Add chunks for both documents
        for doc_id in [self.doc1_id, self.doc2_id]:
            for i in range(1, 3):
                self.chunk_dao.add(Chunk(
                    document_id=doc_id,
                    chunk_text=f"Chunk {i}",
                    file=f"doc{doc_id}.pdf"
                ))
        
        # Delete chunks for both documents
        deleted_count = self.chunk_dao.delete_chunks_by_document_ids([self.doc1_id, self.doc2_id])
        self.assertEqual(deleted_count, 4)
        
        # Verify deletion
        chunks = self.chunk_dao.get_chunks_by_document_ids([self.doc1_id, self.doc2_id])
        self.assertEqual(len(chunks), 0)
    
    def test_fts_search_basic(self):
        """Test basic full-text search"""
        # Add chunks with different content
        chunk1_id = self.chunk_dao.add(Chunk(
            document_id=self.doc1_id,
            chunk_text="Python programming language is powerful and versatile",
            file="doc1.pdf"
        ))
        chunk2_id = self.chunk_dao.add(Chunk(
            document_id=self.doc1_id,
            chunk_text="JavaScript is widely used for web development",
            file="doc1.pdf"
        ))
        chunk3_id = self.chunk_dao.add(Chunk(
            document_id=self.doc1_id,
            chunk_text="Python is great for data science and machine learning",
            file="doc1.pdf"
        ))
        
        # Search for "Python"
        chunk_ids, matched_query = self.chunk_dao.search("Python", limit=10)
        
        self.assertIsNotNone(matched_query)
        self.assertGreater(len(chunk_ids), 0)
        # Both Python chunks should be in results
        self.assertIn(chunk1_id, chunk_ids)
        self.assertIn(chunk3_id, chunk_ids)
    
    def test_fts_search_with_scores(self):
        """Test full-text search with scores"""
        # Add chunks
        chunk1_id = self.chunk_dao.add(Chunk(
            document_id=self.doc1_id,
            chunk_text="Machine learning and artificial intelligence",
            file="doc1.pdf"
        ))
        chunk2_id = self.chunk_dao.add(Chunk(
            document_id=self.doc1_id,
            chunk_text="Deep learning is a subset of machine learning",
            file="doc1.pdf"
        ))
        
        # Search with scores
        chunk_ids, scores, matched_query = self.chunk_dao.search_with_scores(
            "machine learning",
            limit=10
        )
        
        self.assertIsNotNone(matched_query)
        self.assertEqual(len(chunk_ids), len(scores))
        self.assertGreater(len(chunk_ids), 0)
        # Both chunks should be in results
        self.assertIn(chunk1_id, chunk_ids)
        self.assertIn(chunk2_id, chunk_ids)
    
    def test_fts_search_with_subset(self):
        """Test FTS search limited to a subset of chunk IDs"""
        # Add chunks
        chunk1_id = self.chunk_dao.add(Chunk(
            document_id=self.doc1_id,
            chunk_text="Database management systems",
            file="doc1.pdf"
        ))
        chunk2_id = self.chunk_dao.add(Chunk(
            document_id=self.doc1_id,
            chunk_text="Database optimization techniques",
            file="doc1.pdf"
        ))
        chunk3_id = self.chunk_dao.add(Chunk(
            document_id=self.doc2_id,
            chunk_text="Database design patterns",
            file="doc2.pdf"
        ))
        
        # Search only in chunk1 and chunk2
        chunk_ids, matched_query = self.chunk_dao.search(
            "database",
            subset_ids=[chunk1_id, chunk2_id],
            limit=10
        )
        
        self.assertGreater(len(chunk_ids), 0)
        # chunk3 should not be in results
        self.assertNotIn(chunk3_id, chunk_ids)
    
    def test_fts_search_empty_subset(self):
        """Test FTS search with empty subset returns empty results"""
        chunk_ids, matched_query = self.chunk_dao.search(
            "test",
            subset_ids=[],
            limit=10
        )
        
        self.assertEqual(len(chunk_ids), 0)
        self.assertIsNone(matched_query)
    
    def test_fts_search_no_results(self):
        """Test FTS search with no matching results"""
        self.chunk_dao.add(Chunk(
            document_id=self.doc1_id,
            chunk_text="Some random text",
            file="doc1.pdf"
        ))
        
        chunk_ids, matched_query = self.chunk_dao.search(
            "xyzabc123nonexistent",
            limit=10
        )
        
        self.assertEqual(len(chunk_ids), 0)
    
    def test_keyword_search(self):
        """Test keyword search as a phrase"""
        chunk1_id = self.chunk_dao.add(Chunk(
            document_id=self.doc1_id,
            chunk_text="Natural language processing is fascinating",
            file="doc1.pdf"
        ))
        chunk2_id = self.chunk_dao.add(Chunk(
            document_id=self.doc1_id,
            chunk_text="Natural language understanding",
            file="doc1.pdf"
        ))
        
        # Search for exact phrase
        chunk_ids, scores = self.chunk_dao.search_keyword_with_scores(
            "natural language",
            limit=10
        )
        
        self.assertGreater(len(chunk_ids), 0)
        self.assertEqual(len(chunk_ids), len(scores))
    
    def test_keyword_search_with_exclusions(self):
        """Test keyword search with exclusions"""
        chunk1_id = self.chunk_dao.add(Chunk(
            document_id=self.doc1_id,
            chunk_text="Cloud computing infrastructure",
            file="doc1.pdf"
        ))
        chunk2_id = self.chunk_dao.add(Chunk(
            document_id=self.doc1_id,
            chunk_text="Cloud storage solutions",
            file="doc1.pdf"
        ))
        
        # Search but exclude chunk1
        chunk_ids, scores = self.chunk_dao.search_keyword_with_scores(
            "cloud",
            exclude_chunk_ids=[chunk1_id],
            limit=10
        )
        
        # chunk1 should not be in results
        if len(chunk_ids) > 0:
            self.assertNotIn(chunk1_id, chunk_ids)
    
    def test_keyword_search_with_file_filter(self):
        """Test keyword search filtered by file IDs"""
        chunk1_id = self.chunk_dao.add(Chunk(
            document_id=self.doc1_id,
            chunk_text="API design principles",
            file="doc1.pdf"
        ))
        chunk2_id = self.chunk_dao.add(Chunk(
            document_id=self.doc2_id,
            chunk_text="API development best practices",
            file="doc2.pdf"
        ))
        
        # Search only in doc1
        chunk_ids, scores = self.chunk_dao.search_keyword_with_scores(
            "API",
            include_file_ids=[self.doc1_id],
            limit=10
        )
        
        if len(chunk_ids) > 0:
            # chunk2 should not be in results
            self.assertNotIn(chunk2_id, chunk_ids)
    
    def test_fts_triggers_on_insert(self):
        """Test that FTS table is automatically updated on insert"""
        chunk = Chunk(
            document_id=self.doc1_id,
            chunk_text="Testing FTS triggers on insert",
            file="doc1.pdf"
        )
        chunk_id = self.chunk_dao.add(chunk)
        
        # Search should find the newly inserted chunk
        chunk_ids, matched_query = self.chunk_dao.search("triggers", limit=10)
        self.assertIn(chunk_id, chunk_ids)
    
    def test_fts_triggers_on_update(self):
        """Test that FTS table is automatically updated on update"""
        chunk = Chunk(
            document_id=self.doc1_id,
            chunk_text="Original content",
            file="doc1.pdf"
        )
        chunk_id = self.chunk_dao.add(chunk)
        
        # Update the chunk text
        self.chunk_dao.update_by_id(chunk_id, {
            'chunk_text': 'Updated with new searchable content'
        })
        
        # Search should find the updated text
        chunk_ids, matched_query = self.chunk_dao.search("searchable", limit=10)
        self.assertIn(chunk_id, chunk_ids)
        
        # Search for old text should not find it
        chunk_ids_old, _ = self.chunk_dao.search("Original", limit=10)
        self.assertNotIn(chunk_id, chunk_ids_old)
    
    def test_fts_triggers_on_delete(self):
        """Test that FTS table is automatically updated on delete"""
        chunk = Chunk(
            document_id=self.doc1_id,
            chunk_text="Content to be deleted from FTS",
            file="doc1.pdf"
        )
        chunk_id = self.chunk_dao.add(chunk)
        
        # Verify it's searchable
        chunk_ids, _ = self.chunk_dao.search("deleted", limit=10)
        self.assertIn(chunk_id, chunk_ids)
        
        # Delete the chunk
        self.chunk_dao.delete_by_ids([chunk_id])
        
        # Search should not find the deleted chunk
        chunk_ids_after, _ = self.chunk_dao.search("deleted", limit=10)
        self.assertNotIn(chunk_id, chunk_ids_after)
    
    def test_cascade_delete_chunks_on_document_delete(self):
        """Test that chunks are deleted when their parent document is deleted"""
        # Add chunks for doc1
        chunk_ids = []
        for i in range(1, 4):
            chunk_id = self.chunk_dao.add(Chunk(
                document_id=self.doc1_id,
                chunk_text=f"Chunk {i}",
                file="doc1.pdf"
            ))
            chunk_ids.append(chunk_id)
        
        # Delete the document
        self.document_dao.delete_by_ids([self.doc1_id])
        
        # Chunks should be automatically deleted due to foreign key cascade
        remaining_chunks = self.chunk_dao.get_chunks_by_document_id(self.doc1_id)
        self.assertEqual(len(remaining_chunks), 0)
    
    def test_search_with_file_field(self):
        """Test that FTS search works with file field"""
        chunk = Chunk(
            document_id=self.doc1_id,
            chunk_text="Main content about research",
            file="important_research_paper.pdf"
        )
        chunk_id = self.chunk_dao.add(chunk)
        
        # Search by file name
        chunk_ids, _ = self.chunk_dao.search("important research", limit=10)
        self.assertIn(chunk_id, chunk_ids)
        
        # Search by content
        chunk_ids, _ = self.chunk_dao.search("research", limit=10)
        self.assertIn(chunk_id, chunk_ids)


if __name__ == '__main__':
    unittest.main()

