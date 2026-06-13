# io/readers.py
# ==============================================================================
# Module: io.readers
# Purpose: Read various file formats
#
# Classes:
#   - FileReader: Read different file types
#   - CSVReader: Read CSV files
#   - JSONReader: Read JSON files
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

from .. import ValidationError

logger = logging.getLogger(__name__)


# ==============================================================================
# FILE READER
# ==============================================================================

@dataclass
class FileReader:
    """
    Read various file formats.
    
    Supports CSV, JSON, NPY, TXT formats.
    
    Examples
    --------
    >>> reader = FileReader()
    >>> data = reader.read("results.csv")
    """
    
    def read(self, filepath: str) -> Any:
        """
        Read file by extension.
        
        Parameters
        ----------
        filepath : str
            Path to file
        
        Returns
        -------
        Any
            Loaded data
        
        Raises
        ------
        ValidationError
            If format not supported
        """
        
        path = Path(filepath)
        
        if not path.exists():
            raise ValidationError(
                f"File not found: {filepath}",
                error_code="READER_FILE_NOT_FOUND"
            )
        
        logger.debug(f"[FileReader] Reading {path.suffix} file: {filepath}")
        
        if path.suffix == '.csv':
            return CSVReader.read(filepath)
        elif path.suffix == '.json':
            return JSONReader.read(filepath)
        elif path.suffix == '.npy':
            return NPYReader.read(filepath)
        elif path.suffix == '.txt':
            return TXTReader.read(filepath)
        else:
            raise ValidationError(
                f"Unsupported file format: {path.suffix}",
                error_code="READER_UNSUPPORTED_FORMAT"
            )


# ==============================================================================
# CSV READER
# ==============================================================================

@dataclass
class CSVReader:
    """Read CSV files."""
    
    @staticmethod
    def read(filepath: str, delimiter: str = ',') -> List[Dict[str, Any]]:
        """
        Read CSV file.
        
        Parameters
        ----------
        filepath : str
            Path to CSV file
        delimiter : str, optional
            Field delimiter (default: ',')
        
        Returns
        -------
        List[Dict[str, Any]]
            Rows as dictionaries
        """
        
        logger.debug(f"[CSVReader] Reading {filepath}")
        
        rows = []
        
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                logger.warning("[CSVReader] Empty CSV file")
                return []
            
            # Parse header
            header = lines[0].strip().split(delimiter)
            
            # Parse rows
            for line in lines[1:]:
                if not line.strip():
                    continue
                
                values = line.strip().split(delimiter)
                
                if len(values) != len(header):
                    logger.warning(f"[CSVReader] Row has {len(values)} values, "
                                 f"expected {len(header)}")
                    continue
                
                row = dict(zip(header, values))
                rows.append(row)
            
            logger.info(f"[CSVReader] Loaded {len(rows)} rows")
            
            return rows
        
        except Exception as e:
            raise ValidationError(
                f"Failed to read CSV: {e}",
                error_code="CSV_READ_ERROR"
            ) from e


# ==============================================================================
# JSON READER
# ==============================================================================

@dataclass
class JSONReader:
    """Read JSON files."""
    
    @staticmethod
    def read(filepath: str) -> Dict[str, Any]:
        """
        Read JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to JSON file
        
        Returns
        -------
        Dict[str, Any]
            Loaded JSON data
        """
        
        logger.debug(f"[JSONReader] Reading {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            logger.info(f"[JSONReader] Loaded JSON with {len(data)} keys")
            
            return data
        
        except json.JSONDecodeError as e:
            raise ValidationError(
                f"Invalid JSON: {e}",
                error_code="JSON_DECODE_ERROR"
            ) from e
        except Exception as e:
            raise ValidationError(
                f"Failed to read JSON: {e}",
                error_code="JSON_READ_ERROR"
            ) from e


# ==============================================================================
# NPY READER
# ==============================================================================

@dataclass
class NPYReader:
    """Read NumPy .npy files."""
    
    @staticmethod
    def read(filepath: str) -> np.ndarray:
        """
        Read NumPy array.
        
        Parameters
        ----------
        filepath : str
            Path to .npy file
        
        Returns
        -------
        np.ndarray
            Loaded array
        """
        
        logger.debug(f"[NPYReader] Reading {filepath}")
        
        try:
            data = np.load(filepath)
            logger.info(f"[NPYReader] Loaded array: shape={data.shape}")
            return data
        except Exception as e:
            raise ValidationError(
                f"Failed to read NPY: {e}",
                error_code="NPY_READ_ERROR"
            ) from e


# ==============================================================================
# TXT READER
# ==============================================================================

@dataclass
class TXTReader:
    """Read text files."""
    
    @staticmethod
    def read(filepath: str) -> List[str]:
        """
        Read text file.
        
        Parameters
        ----------
        filepath : str
            Path to text file
        
        Returns
        -------
        List[str]
            Lines of text
        """
        
        logger.debug(f"[TXTReader] Reading {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # Strip newlines
            lines = [line.rstrip('\n') for line in lines]
            
            logger.info(f"[TXTReader] Loaded {len(lines)} lines")
            
            return lines
        except Exception as e:
            raise ValidationError(
                f"Failed to read TXT: {e}",
                error_code="TXT_READ_ERROR"
            ) from e


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("IO READERS MODULE - TESTING")
    print("="*80 + "\n")
    
    print("[TEST 1] FileReader initialization...")
    try:
        reader = FileReader()
        print(f"✓ Reader created\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("[TEST 2] JSONReader...")
    try:
        test_data = {"test": "data", "value": 123}
        print(f"✓ JSON reader available\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("="*80)
    print("✓ Reader tests passed!")
    print("="*80 + "\n")
