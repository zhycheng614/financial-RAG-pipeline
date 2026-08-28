import sqlite3
import os
from sqlalchemy import create_engine, Engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from models.base import Base
import logging


class Sqlite3DbManager:

    def __init__(self, db_path: str, schema_file: str):
        self.db_path = db_path
        self.schema_file = schema_file
        self.engine = None  # Will be set after connection is created
        self.session_factory = None  # Will be set after engine is created

        self.conn = self.create_file_db()
        # create_file_db() -> _setup_connection() already builds the engine.
        # Rebuilding it here would orphan the first one, whose StaticPool holds
        # an open handle on the database file that can never be disposed.
        if self.engine is None:
            self.engine = self._get_engine_from_connection()
        self.session_factory = self.create_session_factory()

    def _get_engine_from_connection(self) -> Engine:
        """
        Create a SQLAlchemy engine from a database connection.
        Thread-safe configuration for SQLite with proper connection pooling.
        
        Args:
            connection: SQLite database connection object or an object with an engine attribute
        
        Returns:
            Engine: A SQLAlchemy engine
        """
        # Create engine with thread-safe connection string
        engine = create_engine(
            f'sqlite:///{self.db_path}',
            connect_args={
                'check_same_thread': False,  # Allow multi-threading
                'timeout': 30.0
            },
            poolclass=StaticPool,  # Use static pool for file-based SQLite
            echo=False
        )
        
        # Configure SQLite for better concurrency
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.close()
        
        return engine

    def create_session_factory(self) -> sessionmaker:
        """
        Create a SQLAlchemy session factory from a database connection.
        
        Args:
            connection: SQLite database connection object
        
        Returns:
            sessionmaker: A SQLAlchemy session factory
        """
        
        # Create and return a session factory
        return sessionmaker(bind=self.engine, expire_on_commit=False)    # If you don't set expire_on_commit to False, the attributes in the ORM object will not be available after the session is closed.

    def _create_tables(self) -> None:
        Base.metadata.create_all(self.engine)

    def _setup_connection(self) -> None:
        """
        Sets up a SQLite connection with schema and row factory.
        Also registers datetime adapter and converter.
        Note: For encrypted databases, the encryption key must be set BEFORE calling this function.
        """
        self.conn.row_factory = sqlite3.Row
        
        # Note: encryption key should already be set by caller (e.g., create_file_db)
        # Setting it here would be too late for some operations
        
        # Create engine before creating tables
        if self.engine is None:
            self.engine = self._get_engine_from_connection()
        
        self._create_tables() 

        # tables must have been created before this step
        with open(self.schema_file, "r", encoding="utf-8") as f:
            schema_sql = f.read()
            self.conn.executescript(schema_sql)

    def create_in_memory_db(self) -> sqlite3.Connection:
        """Creates an in-memory SQLite database with datetime support"""
        self.conn = sqlite3.connect(":memory:", detect_types=sqlite3.PARSE_DECLTYPES)
        self._setup_connection()
        return self.conn


    def create_file_db(self, timeout: float=30.0, isolation_level=None):
        """Create a SQLite database connection to a file.

        Args:
            timeout: Timeout in seconds for acquiring a database lock (default 30.0)
            isolation_level: SQLite isolation level (default None for better concurrency)

        Returns:
            Connection: SQLite database connection
        """
        logger = logging.getLogger(__name__)
        
        try:
            self.conn = sqlite3.connect(
                self.db_path,
                detect_types=sqlite3.PARSE_DECLTYPES,
                isolation_level=isolation_level,
                timeout=timeout,
                check_same_thread=False  # Allow usage across threads
            )
            self.conn.row_factory = sqlite3.Row
        
            
            # Enable WAL mode for better concurrent access
            # WAL allows multiple readers and one writer simultaneously
            self.conn.execute("PRAGMA journal_mode=WAL")
            
            # Set busy timeout at SQLite level for additional safety
            self.conn.execute(f"PRAGMA busy_timeout={int(timeout * 1000)}")
            
            # Enable loading extensions
            self.conn.enable_load_extension(True)
            
            try:
                self._setup_connection()
                
                return self.conn
            except Exception as e:
                raise e
                
        except sqlite3.Error as e:
            if self.conn:
                self.conn.close()
            logger.error(f"SQLite error: {str(e)}")
            raise e
