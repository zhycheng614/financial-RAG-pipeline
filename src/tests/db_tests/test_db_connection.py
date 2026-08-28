"""
Test suite for database connection and schema setup
"""
import unittest
import os
import tempfile
import sqlite3
from beans.sqlite3_db_manager import Sqlite3DbManager
from models.document import Document
from models.chunk import Chunk


class TestDbConnection(unittest.TestCase):
    """Test database connection and schema setup"""
    
    def setUp(self):
        """Create a temporary database for testing"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.schema_file = "schemas/fts_schema.sql"
    
    def tearDown(self):
        """Clean up temporary database"""
        if hasattr(self, 'db_manager') and self.db_manager.conn:
            self.db_manager.conn.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_database_creation(self):
        """Test that database file is created"""
        self.db_manager = Sqlite3DbManager(self.db_path, self.schema_file)
        self.assertTrue(os.path.exists(self.db_path))
    
    def test_tables_created(self):
        """Test that all required tables are created"""
        self.db_manager = Sqlite3DbManager(self.db_path, self.schema_file)
        
        # Check that documents table exists
        cursor = self.db_manager.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents'")
        self.assertIsNotNone(cursor.fetchone(), "documents table should exist")
        
        # Check that chunks table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'")
        self.assertIsNotNone(cursor.fetchone(), "chunks table should exist")
    
    def test_fts_tables_created(self):
        """Test that FTS virtual tables are created"""
        self.db_manager = Sqlite3DbManager(self.db_path, self.schema_file)
        
        cursor = self.db_manager.conn.cursor()
        
        # Check chunks_fts table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'")
        self.assertIsNotNone(cursor.fetchone(), "chunks_fts table should exist")
    
    def test_triggers_created(self):
        """Test that FTS triggers are created"""
        self.db_manager = Sqlite3DbManager(self.db_path, self.schema_file)
        
        cursor = self.db_manager.conn.cursor()
        
        # Check for chunks insert trigger
        cursor.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND name='chunks_ai'")
        self.assertIsNotNone(cursor.fetchone(), "chunks_ai trigger should exist")
        
        # Check for chunks delete trigger
        cursor.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND name='chunks_ad'")
        self.assertIsNotNone(cursor.fetchone(), "chunks_ad trigger should exist")
        
        # Check for chunks update trigger
        cursor.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND name='chunks_au'")
        self.assertIsNotNone(cursor.fetchone(), "chunks_au trigger should exist")
    
    def test_wal_mode_enabled(self):
        """Test that WAL mode is enabled for better concurrency"""
        self.db_manager = Sqlite3DbManager(self.db_path, self.schema_file)
        
        cursor = self.db_manager.conn.cursor()
        cursor.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        self.assertEqual(mode.lower(), "wal", "WAL mode should be enabled")
    
    def test_session_factory_created(self):
        """Test that SQLAlchemy session factory is created"""
        self.db_manager = Sqlite3DbManager(self.db_path, self.schema_file)
        self.assertIsNotNone(self.db_manager.session_factory, "Session factory should be created")
        
        # Test that we can create a session
        session = self.db_manager.session_factory()
        self.assertIsNotNone(session)
        session.close()
    
    def test_engine_created(self):
        """Test that SQLAlchemy engine is created"""
        self.db_manager = Sqlite3DbManager(self.db_path, self.schema_file)
        self.assertIsNotNone(self.db_manager.engine, "Engine should be created")
    
    def test_connection_persistence(self):
        """Test that connection persists and can be used multiple times"""
        self.db_manager = Sqlite3DbManager(self.db_path, self.schema_file)
        
        # Execute multiple queries
        cursor = self.db_manager.conn.cursor()
        cursor.execute("SELECT 1")
        self.assertEqual(cursor.fetchone()[0], 1)
        
        cursor.execute("SELECT 2")
        self.assertEqual(cursor.fetchone()[0], 2)


if __name__ == '__main__':
    unittest.main()

