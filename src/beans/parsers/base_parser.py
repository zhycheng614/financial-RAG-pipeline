from abc import ABC, abstractmethod
import os
import logging

logger = logging.getLogger(__name__)


class BaseParser(ABC):
    """Base class for all document parsers.
    
    Parsers are responsible for extracting content from files.
    They should be pure - only parsing, no chunking or processing.
    Chunking is handled separately by the Chunker class.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

    @abstractmethod
    def parse(self):
        """Parse the document and return parsed content.
        
        The return type depends on the parser implementation.
        For PDFParser, returns List[PDFPage].
        """
        pass