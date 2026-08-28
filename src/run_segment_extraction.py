#!/usr/bin/env python3
"""
Convenience script to run the segment extraction pipeline.

Usage:
    python run_segment_extraction.py --pdf-folder data/10k --queries-csv data/segments_queries/Segment_Q.csv
    
    # With custom model
    python run_segment_extraction.py --pdf-folder data/10k --queries-csv data/segments_queries/Segment_Q.csv --model gpt-4o-mini
"""

from main.main_segment_extraction_pipeline import main

if __name__ == "__main__":
    main()

