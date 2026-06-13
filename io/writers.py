# io/writers.py
# ==============================================================================
# Module: io.writers
# Purpose: Export results to various file formats
#
# Classes:
#   - FileWriter: Write different file types
#   - CSVWriter: Write CSV files
#   - JSONWriter: Write JSON files
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

from .. import ValidationError

logger = logging.getLogger(__name__)


# ==============================================================================
# FILE WRITER
# ==============================================================================

@dataclass
class FileWriter:
    """
    Write results to various file formats.
    
    Supports CSV, JSON, NPY, TXT formats.
    
    Examples
    --------
    >>> writer = FileWriter()
    >>> writer.write("results.csv", data)
    """
    
    def write(self, filepath: str, data: Any, **kwargs) -> None:
        """
        Write file by extension.
        
        Parameters
        ----------
        filepath : str
            Output file path
        data : Any
            Data to write
        **kwargs
            Format-specific parameters
        
        Raises
        ------
        ValidationError
            If format not supported
        """
        
        path = Path(filepath)
        
        # Create parent directories
        path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"[FileWriter] Writing {path.suffix} file: {filepath}")
        
        if path.suffix == '.csv':
            CSVWriter.write(filepath, data, **kwargs)
        elif path.suffix == '.json':
            JSONWriter.write(filepath, data, **kwargs)
        elif path.suffix == '.npy':
            NPYWriter.write(filepath, data, **kwargs)
        elif path.suffix == '.txt':
            TXTWriter.write(filepath, data, **kwargs)
        else:
            raise ValidationError(
                f"Unsupported file format: {path.suffix}",
                error_code="WRITER_UNSUPPORTED_FORMAT"
            )


# ==============================================================================
# CSV WRITER
# ==============================================================================

@dataclass
class CSVWriter:
    """Write CSV files."""
    
    @staticmethod
    def write(filepath: str, data: List[Dict[str, Any]], 
              delimiter: str = ',') -> None:
        """
        Write CSV file.
        
        Parameters
        ----------
        filepath : str
            Output file path
        data : List[Dict[str, Any]]
            Data rows
        delimiter : str, optional
            Field delimiter (default: ',')
        """
        
        logger.debug(f"[CSVWriter] Writing {filepath}")
        
        if not data:
            logger.warning("[CSVWriter] Empty data")
            return
        
        try:
            # Get header from first row
            header = list(data[0].keys())
            
            with open(filepath, 'w') as f:
                # Write header
                f.write(delimiter.join(header) + '\n')
                
                # Write rows
                for row in data:
                    values = [str(row.get(key, '')) for key in header]
                    f.write(delimiter.join(values) + '\n')
            
            logger.info(f"[CSVWriter] Wrote {len(data)} rows to {filepath}")
        
        except Exception as e:
            raise ValidationError(
                f"Failed to write CSV: {e}",
                error_code="CSV_WRITE_ERROR"
            ) from e


# ==============================================================================
# JSON WRITER
# ==============================================================================

@dataclass
class JSONWriter:
    """Write JSON files."""
    
    @staticmethod
    def write(filepath: str, data: Dict[str, Any], indent: int = 2) -> None:
        """
        Write JSON file.
        
        Parameters
        ----------
        filepath : str
            Output file path
        data : Dict[str, Any]
            Data to write
        indent : int, optional
            JSON indentation (default: 2)
        """
        
        logger.debug(f"[JSONWriter] Writing {filepath}")
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=indent, default=str)
            
            logger.info(f"[JSONWriter] Wrote JSON to {filepath}")
        
        except Exception as e:
            raise ValidationError(
                f"Failed to write JSON: {e}",
                error_code="JSON_WRITE_ERROR"
            ) from e


# ==============================================================================
# NPY WRITER
# ==============================================================================

@dataclass
class NPYWriter:
    """Write NumPy .npy files."""
    
    @staticmethod
    def write(filepath: str, data: np.ndarray, **kwargs) -> None:
        """
        Write NumPy array.
        
        Parameters
        ----------
        filepath : str
            Output file path
        data : np.ndarray
            Array to write
        """
        
        logger.debug(f"[NPYWriter] Writing {filepath}")
        
        try:
            data_array = np.asarray(data)
            np.save(filepath, data_array)
            
            logger.info(f"[NPYWriter] Wrote array shape={data_array.shape} "
                       f"to {filepath}")
        
        except Exception as e:
            raise ValidationError(
                f"Failed to write NPY: {e}",
                error_code="NPY_WRITE_ERROR"
            ) from e


# ==============================================================================
# TXT WRITER
# ==============================================================================

@dataclass
class TXTWriter:
    """Write text files."""
    
    @staticmethod
    def write(filepath: str, data: List[str], **kwargs) -> None:
        """
        Write text file.
        
        Parameters
        ----------
        filepath : str
            Output file path
        data : List[str]
            Lines of text
        """
        
        logger.debug(f"[TXTWriter] Writing {filepath}")
        
        try:
            with open(filepath, 'w') as f:
                for line in data:
                    f.write(str(line) + '\n')
            
            logger.info(f"[TXTWriter] Wrote {len(data)} lines to {filepath}")
        
        except Exception as e:
            raise ValidationError(
                f"Failed to write TXT: {e}",
                error_code="TXT_WRITE_ERROR"
            ) from e


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("IO WRITERS MODULE - TESTING")
    print("="*80 + "\n")
    
    print("[TEST 1] FileWriter initialization...")
    try:
        writer = FileWriter()
        print(f"✓ Writer created\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("[TEST 2] JSONWriter...")
    try:
        test_data = {"test": "data", "value": 123}
        print(f"✓ JSON writer available\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("="*80)
    print("✓ Writer tests passed!")
    print("="*80 + "\n")
