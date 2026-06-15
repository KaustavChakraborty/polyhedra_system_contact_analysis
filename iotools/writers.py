# iotools/writers.py
# ==============================================================================
# Module: iotools.writers
# Purpose: Export results to various file formats
#
# Classes:
#   - FileWriter: Write different file types
#   - CSVWriter: Write CSV files
#   - JSONWriter: Write JSON files
#
# Main control flow:
#
#     FileWriter.write(filepath, data)
#         |
#         |-- inspect file extension
#         |-- .csv  -> CSVWriter.write(...)
#         |-- .json -> JSONWriter.write(...)
#         |-- .npy  -> NPYWriter.write(...)
#         |-- .txt  -> TXTWriter.write(...)
#         `-- otherwise raise ValidationError
#
# Author: Kaustav Chakraborty
# ==============================================================================

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

# Fix relative imports
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import ValidationError

logger = logging.getLogger(__name__)



# ==============================================================================
# Helper functions
# ==============================================================================

def _ensure_parent_directory(filepath: str | Path) -> Path:
    """
    Normalize an output path and create its parent directory if necessary.

    Example
    -------
    If filepath is:

        results/contact_metrics/frame_0001.json

    then this helper ensures that:

        results/contact_metrics/

    exists before writing begins.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_text_lines(data: Any) -> List[str]:
    """
    Convert text-like input into a list of strings.

    TXT output is often called with different forms of input:

        - one long string
        - list of lines
        - tuple/generator of values
        - single numerical value

    This helper standardizes those cases so TXTWriter can write line by line.
    """
    if isinstance(data, str):
        # splitlines() avoids double newlines if the input already has them.
        return data.splitlines() or [data]

    if isinstance(data, Iterable):
        return [str(item) for item in data]

    return [str(data)]


# ==============================================================================
# FILE WRITER
# ==============================================================================

@dataclass
class FileWriter:
    """
    General file writer that dispatches based on filename extension.

    This is the class most other modules should use. The caller only needs to
    know the desired output filename; this class selects the correct specialized
    writer.

    Examples
    --------
    >>> writer = FileWriter()
    >>> writer.write("results/summary.json", {"frames": 100})
    >>> writer.write("results/table.csv", [{"a": 1, "b": 2}])
    >>> writer.write("results/array.npy", np.array([1, 2, 3]))
    >>> writer.write("results/report.txt", ["line 1", "line 2"])
    """
    
    def write(self, filepath: str, data: Any, **kwargs) -> None:
        """
        Write data to a file selected by file extension.

        Parameters
        ----------
        filepath
            Output path. The suffix decides which writer is used.

        data
            The object to write.

        **kwargs
            Additional format-specific options. For example:

                delimiter=";" for CSVWriter
                indent=4 for JSONWriter

        Raises
        ------
        ValidationError
            If the extension is unsupported or the specialized writer fails.
        """
        
        path = _ensure_parent_directory(filepath)
        suffix = path.suffix.lower()

        logger.debug("[FileWriter] Writing %s file: %s", suffix, path)

        if suffix == ".csv":
            CSVWriter.write(path, data, **kwargs)
        elif suffix == ".json":
            JSONWriter.write(path, data, **kwargs)
        elif suffix == ".npy":
            NPYWriter.write(path, data, **kwargs)
        elif suffix == ".txt":
            TXTWriter.write(path, data, **kwargs)
        else:
            raise ValidationError(
                f"Unsupported file format: {path.suffix}",
                error_code="WRITER_UNSUPPORTED_FORMAT",
            )


# ==============================================================================
# CSV WRITER
# ==============================================================================

@dataclass
class CSVWriter:
    """Write CSV files."""
    
    @staticmethod
    def write(filepath: str | Path, data: Sequence[Dict[str, Any]], delimiter: str = ",", fieldnames: Sequence[str] | None = None, write_empty_file: bool = False) -> None:
        """
        Write rows to a CSV file.

        Parameters
        ----------
        filepath
            Output CSV path.

        data
            Sequence of dictionaries. Each dictionary is one row.

        delimiter
            CSV delimiter. Default is comma.

        fieldnames
            Optional explicit column order. If not supplied, keys from the first
            row are used.

        write_empty_file
            If True, create an empty file when data is empty. If False, do not
            write anything for empty data.
        """
        path = _ensure_parent_directory(filepath)
        logger.debug("[CSVWriter] Writing %s", path)

        if data is None:
            raise ValidationError(
                "CSVWriter received data=None; expected a sequence of dictionaries.",
                error_code="CSV_WRITE_NONE_DATA",
            )

        if len(data) == 0:
            logger.warning("[CSVWriter] Empty data for %s", path)
            if write_empty_file:
                path.write_text("", encoding="utf-8")
            return

        if not isinstance(data[0], dict):
            raise ValidationError(
                "CSVWriter expects a sequence of dictionaries.",
                error_code="CSV_WRITE_INVALID_DATA",
            )

        if fieldnames is None:
            fieldnames = list(data[0].keys())

        try:
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=list(fieldnames),
                    delimiter=delimiter,
                    extrasaction="ignore",
                )
                writer.writeheader()
                for row in data:
                    writer.writerow(row)

            logger.info("[CSVWriter] Wrote %d rows to %s", len(data), path)

        except Exception as exc:
            raise ValidationError(
                f"Failed to write CSV: {exc}",
                error_code="CSV_WRITE_ERROR",
            ) from exc


# ==============================================================================
# JSON WRITER
# ==============================================================================

@dataclass
class JSONWriter:
    """Write JSON files."""
    
    @staticmethod
    def write(
        filepath: str | Path,
        data: Any,
        indent: int = 2,
        sort_keys: bool = False,
    ) -> None:
        """Write data to a JSON file."""
        path = _ensure_parent_directory(filepath)
        logger.debug("[JSONWriter] Writing %s", path)

        try:
            with path.open("w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=indent, sort_keys=sort_keys, default=str)

            logger.info("[JSONWriter] Wrote JSON to %s", path)

        except Exception as exc:
            raise ValidationError(
                f"Failed to write JSON: {exc}",
                error_code="JSON_WRITE_ERROR",
            ) from exc


# ==============================================================================
# NPY WRITER
# ==============================================================================

@dataclass
class NPYWriter:
    """Write NumPy .npy files."""
    
    @staticmethod
    def write(filepath: str | Path, data: Any, **kwargs: Any) -> None:
        """Save array-like data using numpy.save."""
        path = _ensure_parent_directory(filepath)
        logger.debug("[NPYWriter] Writing %s", path)

        try:
            data_array = np.asarray(data)
            np.save(path, data_array, **kwargs)
            logger.info("[NPYWriter] Wrote array shape=%s to %s", data_array.shape, path)

        except Exception as exc:
            raise ValidationError(
                f"Failed to write NPY: {exc}",
                error_code="NPY_WRITE_ERROR",
            ) from exc


# ==============================================================================
# TXT WRITER
# ==============================================================================

@dataclass
class TXTWriter:
    """Write text files."""
    
    @staticmethod
    def write(filepath: str | Path, data: Any, **kwargs: Any) -> None:
        """Write text-like data line by line."""
        path = _ensure_parent_directory(filepath)
        logger.debug("[TXTWriter] Writing %s", path)

        try:
            lines = _normalize_text_lines(data)
            with path.open("w", encoding="utf-8") as handle:
                for line in lines:
                    handle.write(str(line) + "\n")

            logger.info("[TXTWriter] Wrote %d lines to %s", len(lines), path)

        except Exception as exc:
            raise ValidationError(
                f"Failed to write TXT: {exc}",
                error_code="TXT_WRITE_ERROR",
            ) from exc


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("IO WRITERS MODULE - TESTING")
    print("="*80 + "\n")
    
    print("[TEST 1] FileWriter initialization...")
    try:
        writer = FileWriter()
        print(f"[SUCCESS] Writer created\n")
    except Exception as e:
        print(f"[FAILED] Error: {e}\n")
    
    print("[TEST 2] JSONWriter...")
    try:
        test_data = {"test": "data", "value": 123}
        print(f"[SUCCESS] JSON writer available\n")
    except Exception as e:
        print(f"[FAILED] Error: {e}\n")
    
    print("="*80)
    print("[SUCCESS] Writer tests passed!")
    print("="*80 + "\n")
