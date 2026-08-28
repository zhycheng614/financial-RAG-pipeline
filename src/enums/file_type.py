from enum import Enum

class FileType(Enum):
    PDF = "pdf"
    OTHER = "other"

    @classmethod
    def from_filename(cls, filename: str):
        """Determine file type based on filename extension.

        Args:
            filename: Path to the file

        Returns:
            FileType: The detected file type enum value
        """
        if not filename:
            return cls.OTHER

        # Extract extension (handling case with or without the dot)
        extension = filename.split('.')[-1].lower() if '.' in filename else ''

        # Check against each enum value
        for file_type in cls:
            if extension == file_type.value:
                return file_type

        # Default to OTHER if no match found
        return cls.OTHER