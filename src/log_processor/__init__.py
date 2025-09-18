"""
ATM Log Processor Package

This package provides functionality to read, parse, and process ATM log files
for use in RAG (Retrieval Augmented Generation) systems.

Components:
- log_reader: Read JSON log files
- log_parser: Parse and extract fields from log entries
- text_extractor: Convert structured log data to text for embeddings
- validator: Validate required fields in log entries
"""

from .log_reader import LogReader
from .log_parser import LogParser
from .text_extractor import TextExtractor
from .validator import LogValidator

__all__ = ['LogReader', 'LogParser', 'TextExtractor', 'LogValidator']