#!/usr/bin/env python
"""
Convenience script to run the indexing pipeline from project root.
Automatically sets up Python path and runs main_index.py.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run main_index
from main import main_index

if __name__ == "__main__":
    main_index.main()

