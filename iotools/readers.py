# iotools/readers.py
# ==============================================================================
# Module: iotools.readers
# PURPOSE OF THIS MODULE
# ----------------------
# This module centralizes simple file-reading operations for the project.
# Instead of scattering code like this throughout the analysis pipeline:
#
#     with open(path) as f:
#         data = json.load(f)
#
# or:
#
#     arr = np.load(path)
#
# the workflow can call reader classes from one place.
#
# Main classes
# ------------
#
#     FileReader
#         Generic dispatcher. It checks the file extension and calls the correct
#         specialized reader.
#
#     CSVReader
#         Reads CSV files and returns a list of dictionaries.
#
#     JSONReader
#         Reads JSON files and returns the parsed Python object.
#
#     NPYReader
#         Reads NumPy .npy binary arrays.
#
#     TXTReader
#         Reads plain text files and returns a list of lines.
#
# Control-flow idea
# -----------------
#
#     FileReader().read("something.json")
#       └── sees suffix .json
#             └── JSONReader.read("something.json")
#                   └── json.load(...)
#
# So FileReader is a front door. The specialized readers do the actual work.
#
# Author: Kaustav Chakraborty
# ==============================================================================

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

# Fix relative imports
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from core import ValidationError

logger = logging.getLogger(__name__)


# ==============================================================================
# GENERIC FILE READER
# ==============================================================================

@dataclass
class FileReader:
    """
    Generic file reader that dispatches based on filename extension.

    This class does not itself know how to parse CSV, JSON, NPY, or TXT data.
    Instead, it performs these steps:

        1. Convert the input string to a ``Path`` object.
        2. Check that the path exists and is a file.
        3. Inspect the suffix/extension.
        4. Call the matching specialized reader.

    Example
    -------
    >>> reader = FileReader()
    >>> config = reader.read("configs/config_default.json")
    >>> rows = reader.read("results/summary.csv")

    Supported extensions
    --------------------
    .csv
        Uses CSVReader.

    .json
        Uses JSONReader.

    .npy
        Uses NPYReader.

    .txt
        Uses TXTReader.
    """
    
    def read(self, filepath: str) -> Any:
        """
        Read a file using the appropriate specialized reader.

        Parameters
        ----------
        filepath : str | pathlib.Path
            File path to read.

        Returns
        -------
        Any
            The return type depends on the file extension:

                .csv  -> list[dict[str, Any]]
                .json -> Python object, usually dict or list
                .npy  -> numpy.ndarray
                .txt  -> list[str]

        Raises
        ------
        ValidationError
            If the file does not exist, is not a regular file, or has an
            unsupported extension.
        """
        
        path = Path(filepath).expanduser()

        if not path.is_file():
            raise ValidationError(
                f"File not found: {path}",
                error_code="READER_FILE_NOT_FOUND",
            )

        # Normalize extension to lower case so .JSON and .json behave the same.
        suffix = path.suffix.lower()

        logger.debug("[FileReader] Dispatching %s file: %s", suffix, path)

        if suffix == ".csv":
            return CSVReader.read(path)

        if suffix == ".json":
            return JSONReader.read(path)

        if suffix == ".npy":
            return NPYReader.read(path)

        if suffix == ".txt":
            return TXTReader.read(path)

        raise ValidationError(
            f"Unsupported file format: {suffix}",
            error_code="READER_UNSUPPORTED_FORMAT",
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
        
        path = Path(filepath).expanduser()
        logger.debug("[CSVReader] Reading %s", path)

        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter=delimiter)

                # DictReader uses the first non-empty line as column names.
                # If the file is empty, fieldnames will be None.
                if reader.fieldnames is None:
                    logger.warning("[CSVReader] Empty CSV file: %s", path)
                    return []

                rows: List[Dict[str, Any]] = []

                for row_number, row in enumerate(reader, start=2):
                    # csv.DictReader returns None keys when a row has more fields
                    # than the header. That usually means malformed CSV.
                    if None in row:
                        logger.warning(
                            "[CSVReader] Row %d has extra columns beyond header in %s; skipping row",
                            row_number,
                            path,
                        )
                        continue

                    rows.append(dict(row))

        except OSError as exc:
            raise ValidationError(
                f"Failed to read CSV file {path}: {exc}",
                error_code="CSV_READ_ERROR",
            ) from exc

        except Exception as exc:
            raise ValidationError(
                f"Failed to parse CSV file {path}: {exc}",
                error_code="CSV_READ_ERROR",
            ) from exc

        logger.info("[CSVReader] Loaded %d rows from %s", len(rows), path)
        return rows


# ==============================================================================
# JSON READER
# ==============================================================================

@dataclass
class JSONReader:
    """Read JSON files."""
    
    @staticmethod
    def read(filepath: str) -> Dict[str, Any]:
        """
        Read and parse a JSON file.

        Parameters
        ----------
        filepath : str | pathlib.Path
            Path to the JSON file.

        Returns
        -------
        Any
            Parsed JSON object. Usually this is a dict, but JSON also supports
            lists, strings, numbers, booleans, and null.

        Raises
        ------
        ValidationError
            If the file cannot be read or the JSON syntax is invalid.
        """

        path = Path(filepath).expanduser()
        logger.debug("[JSONReader] Reading %s", path)

        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)

        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"Invalid JSON in {path}: line {exc.lineno}, column {exc.colno}: {exc.msg}",
                error_code="JSON_DECODE_ERROR",
            ) from exc

        except OSError as exc:
            raise ValidationError(
                f"Failed to read JSON file {path}: {exc}",
                error_code="JSON_READ_ERROR",
            ) from exc

        except Exception as exc:
            raise ValidationError(
                f"Failed to parse JSON file {path}: {exc}",
                error_code="JSON_READ_ERROR",
            ) from exc

        # len(data) works for dict/list/string but not for numbers/None.
        if hasattr(data, "__len__"):
            logger.info("[JSONReader] Loaded JSON from %s with length/key-count %d", path, len(data))
        else:
            logger.info("[JSONReader] Loaded scalar JSON from %s", path)

        return data


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
        
        path = Path(filepath).expanduser()
        logger.debug("[NPYReader] Reading %s", path)

        try:
            # allow_pickle=False is safer. It prevents loading arbitrary Python
            # objects stored inside .npy files. For pure numeric arrays, this is
            # the preferred setting.
            data = np.load(path, allow_pickle=False)

        except Exception as exc:
            raise ValidationError(
                f"Failed to read NPY file {path}: {exc}",
                error_code="NPY_READ_ERROR",
            ) from exc

        logger.info("[NPYReader] Loaded array from %s with shape=%s dtype=%s", path, data.shape, data.dtype)
        return data


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
        
        path = Path(filepath).expanduser()
        logger.debug("[TXTReader] Reading %s", path)

        try:
            with path.open("r", encoding="utf-8") as handle:
                lines = [line.rstrip("\n") for line in handle]

        except OSError as exc:
            raise ValidationError(
                f"Failed to read TXT file {path}: {exc}",
                error_code="TXT_READ_ERROR",
            ) from exc

        except Exception as exc:
            raise ValidationError(
                f"Failed to parse TXT file {path}: {exc}",
                error_code="TXT_READ_ERROR",
            ) from exc

        logger.info("[TXTReader] Loaded %d lines from %s", len(lines), path)
        return lines

# ==============================================================================
# DIRECT MODULE TEST
# ==============================================================================

if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("IO READERS MODULE - TESTING")
    print("="*80 + "\n")
    
    print("[TEST 1] FileReader initialization...")
    try:
        reader = FileReader()
        print(f"[SUCCESSS] Reader created\n")
    except Exception as e:
        print(f"[FAILURE] Error: {e}\n")
    
    print("[TEST 2] JSONReader...")
    try:
        test_data = {"test": "data", "value": 123}
        print(f"[SUCCESSS] JSON reader available\n")
    except Exception as e:
        print(f"[FAILURE] Error: {e}\n")
    
    print("="*80)
    print("[SUCCESSS] Reader tests passed!")
    print("="*80 + "\n")
