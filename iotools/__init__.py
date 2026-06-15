# io/__init__.py
# ==============================================================================
# Module: io
# Purpose: Input/Output domain - configuration, reading, writing, reporting
#
# Exports:
#   - ConfigLoader (configuration)
#   - FileReader, CSVReader, JSONReader, NPYReader, TXTReader (readers)
#   - FileWriter, CSVWriter, JSONWriter, NPYWriter, TXTWriter (writers)
#   - ReportGenerator, SummaryGenerator (reporters)
#
# Author: Kaustav Chakraborty
# ==============================================================================

from .config import ConfigLoader
from .readers import FileReader, CSVReader, JSONReader, NPYReader, TXTReader
from .writers import FileWriter, CSVWriter, JSONWriter, NPYWriter, TXTWriter
from .reporters import ReportGenerator, SummaryGenerator

__all__ = [
    # Configuration
    'ConfigLoader',
    
    # Readers
    'FileReader',
    'CSVReader',
    'JSONReader',
    'NPYReader',
    'TXTReader',
    
    # Writers
    'FileWriter',
    'CSVWriter',
    'JSONWriter',
    'NPYWriter',
    'TXTWriter',
    
    # Reporters
    'ReportGenerator',
    'SummaryGenerator',
]

__version__ = '2.0.0'
__author__ = 'Kaustav Chakraborty'
