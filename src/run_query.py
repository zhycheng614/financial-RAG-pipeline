#!/usr/bin/env python
"""
Convenience script to run the query pipeline from project root.
Automatically sets up Python path and runs main_query.py.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run main_query
from main import main_query

if __name__ == "__main__":
    asyncio.run(main_query.main())

