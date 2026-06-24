"""
Shape-JSON loading helpers.

Overview
--------
This module provides utilities to load particle shape geometry data from
JSON files. It handles file I/O and JSON parsing with minimal processing.

What This Module Does
---------------------
This module performs low-level file I/O operations:

    1. Open shape JSON file
    2. Parse JSON content
    3. Extract vertex coordinate data
    4. Return raw data structures

Data Preservation
-----------------
The functions preserve the exact reference workflow's shape-loading behavior:

    # Reference behavior:
    with open(shape_file) as handle:
        data = json.load(handle)
    vertices = data["8_vertices"]

This module replicates this behavior exactly, with no changes or improvements.

Usage Context
-------------
These functions are typically called from the workflow layer:

    from workflow import prepare_shape_geometry

    # Which internally calls:
    from dataio import load_shape_vertices

    vertices = load_shape_vertices("shape_Cube.json")

The workflow layer provides error handling and user-friendly messages.

This allows callers to pass either:
    - String paths: "shape.json" or "/path/to/shape.json"
    - Path objects: Path("shape.json")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union

# Type alias for path parameters
# Allows functions to accept either string or Path objects
PathLike = Union[str, Path]


def load_shape_data(shape_file: PathLike) -> Dict[str, Any]:
    """
    Load and return the complete shape JSON dictionary.

    This function reads the shape JSON file and returns the entire
    parsed dictionary. All data from the file is made available.

    Parameters
    ----------
    shape_file : str or pathlib.Path
        Path to the particle-shape JSON file. Can be:
            - Absolute path: "/home/user/shape.json"
            - Relative path: "shape.json" or "data/shape.json"
            - Path object: Path("shape.json")

    Raises
    ------
    FileNotFoundError
        If the shape_file does not exist or cannot be opened.
        This is intentionally allowed to propagate from the built-in
        open() function.

        Example error: "No such file or directory: 'shape.json'"

    json.JSONDecodeError
        If the shape file contains invalid JSON syntax.
        This is intentionally allowed to propagate from json.load().

        Contains information about:
            - Line number where error occurred
            - Column number where error occurred
            - Description of the JSON syntax error

        Example error: "Expecting value: line 5 column 2 (char 42)"

    UnicodeDecodeError
        If the file contains invalid UTF-8 characters.
        This can occur if the file is saved in a different encoding.

        Example error: "'utf-8' codec can't decode byte 0xff in position 0"

    Notes
    -----
    - The function does NOT validate the structure of the returned dict
    - It does NOT check that required fields exist (e.g., "8_vertices")
    - It returns the raw JSON data without any processing
    - For typical usage, see load_shape_vertices() which extracts vertices
    """

    # Open the file in read mode with UTF-8 encoding
    with open(shape_file, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_shape_vertices(shape_file: PathLike) -> List[List[float]]:
    """
    Load the reference particle vertices from the shape JSON file.

    This preserves the exact field used by the reference workflow:

        data["8_vertices"]

    Returns
    -------
    list of list
        Shape vertices in the same list-based structure stored in the JSON.

    Raises
    ------
    KeyError
        If the JSON does not contain the required ``"8_vertices"`` field.

    Notes
    -----
    - This function is hardcoded to look for "8_vertices"
    - If your shape has a different vertex count, use load_shape_data()
      directly and access the appropriate field
    - The returned structure is a list of lists, not a numpy array
      (compatible with the reference workflow behavior)
    - No scaling or transformation is applied to the coordinates
    - Vertices are returned in the same order as stored in the file
    """

    # Load the complete shape data using load_shape_data()
    # This handles file I/O and JSON parsing
    data = load_shape_data(shape_file)

    # Extract and return the vertex
    # KeyError propagates if the field doesn't exist
    # This is intentional; the caller should handle it or see the error
    return data["8_vertices"]
