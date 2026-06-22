"""
JSON loading helpers for the contact-analysis workflow.

Design Philosophy
-----------------
- This module handles ONLY file I/O and JSON parsing.
- It does NOT validate parameter semantics or values.
- Validation is separated into the validation.py module.
- Each function raises ConfigError on failure with descriptive messages.

Error Handling
--------------
All functions in this module wrap low-level I/O and JSON parsing errors
in ConfigError exceptions. This allows the caller to handle all
configuration problems uniformly, without worrying about multiple
exception types.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

from .errors import ConfigError


def load_json_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load the parameter JSON file.

    This function reads the main configuration file that defines all
    parameters for the contact-analysis workflow. The file is expected
    to be a single JSON object (dictionary) containing key-value pairs
    for each parameter.

    The function handles:
        - Path resolution (string or Path objects)
        - File existence checking
        - JSON syntax parsing and error reporting
        - Type validation (must be a dictionary, not a list or scalar)
        - Unicode/UTF-8 encoding handling

    Parameters
    ----------
    config_path : str or pathlib.Path
        Path to the JSON parameter file. Can be:
            - An absolute path (e.g., "/home/user/param_file.json")
            - A relative path (e.g., "param_file.json")
            - A path with tilde expansion (e.g., "~/config/param_file.json")
        The path is converted to a Path object for cross-platform compatibility.

    Returns
    -------
    dict
        Parsed JSON dictionary containing all configuration parameters.
        Each key-value pair represents one configuration option.
        Example structure:
            {
                "shape_name": "Cube",
                "num_edges": 12,
                "num_faces": 6,
                "analysis_path": "/path/to/analysis",
                ...
            }

    Raises
    ------
    ConfigError
        Raised in several error conditions:

        1. File does not exist:
           "Parameter file not found: {config_path}"

        2. Invalid JSON syntax:
           "Invalid JSON syntax in {config_path}: line {N}, column {M}. ..."

        3. File cannot be read (permission error, encoding issue, etc.):
           "Could not read parameter file {config_path}: {original_error}"

        4. JSON is not a dictionary (e.g., a list or scalar value):
           "Expected {config_path} to contain a JSON object/dictionary, ..."


    Example
    -------
    Load a configuration file and catch errors:

        try:
            config = load_json_config("param_file.json")
            print(f"Loaded {len(config)} parameters")
        except ConfigError as exc:
            print(f"Failed to load config: {exc}")
            sys.exit(2)
    """

    # Convert string or Path to Path object for uniform handling
    config_path = Path(config_path)

    # Check that the file exists before attempting to open it
    if not config_path.is_file():
        raise ConfigError(f"Parameter file not found: {config_path}")

    # Read and parse the JSON file
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            # Parse JSON from the file handle
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        # JSONDecodeError includes line and column information
        raise ConfigError(
            f"Invalid JSON syntax in {config_path}: line {exc.lineno}, "
            f"column {exc.colno}. Original error: {exc.msg}"
        ) from exc
    except OSError as exc:
        # OSError covers file permission errors, encoding errors, etc.
        raise ConfigError(f"Could not read parameter file {config_path}: {exc}") from exc

    # Ensure the loaded data is a dictionary, not a list, string, or other type
    # The JSON file MUST contain a single object, not an array or scalar
    if not isinstance(data, dict):
        raise ConfigError(
            f"Expected {config_path} to contain a JSON object/dictionary, "
            f"but got {type(data).__name__}."
        )

    return data


def load_metric_definitions(config_path: Union[str, Path] = "metric_definitions.json") -> Dict[str, str]:
    """
    Load the metric definitions mapping file.

    This function loads the metric_definitions.json file which contains
    human-readable descriptions of distance metrics used in the contact
    analysis. Each metric is identified by a numeric index (0, 1, 2, ...)
    and mapped to a descriptive name.

    The metric definitions file is used to:
        1. Validate that the metric indices requested in the main config exist
        2. Provide human-readable output when printing configuration summaries
        3. Allow the workflow to be self-documenting about what each metric means

    File Format
    -----------
    The metric_definitions.json file must contain a JSON object with a single
    key "metric_definitions" that maps to a dictionary. Example:

        {
            "metric_definitions": {
                "0": "face_center_to_face_center",
                "1": "face_center_to_edge",
                "2": "edge_to_edge",
                "3": "vertex_to_vertex",
                ...
            }
        }

    Parameters
    ----------
    config_path : str or pathlib.Path, default "metric_definitions.json"
        Path to the metric definitions JSON file. If not provided, defaults
        to "metric_definitions.json" in the current working directory.
        Can be:
            - An absolute path (e.g., "/path/to/metric_definitions.json")
            - A relative path (e.g., "metric_definitions.json")
            - A path with tilde expansion (though not expanded by this function)

    Returns
    -------
    dict
        Dictionary mapping metric-index strings to metric names/descriptions.
        Keys are string representations of indices (e.g., "0", "1", "2").
        Values are human-readable metric names (e.g., "face_center_to_face_center").

        Example return value:
            {
                "0": "face_center_to_face_center",
                "1": "face_center_to_edge",
                "2": "edge_to_edge",
                "3": "vertex_to_vertex"
            }

    """

    # Convert to Path object for uniform handling
    metric_file = Path(config_path)

    # Check that the file exists before attempting to open it
    if not metric_file.is_file():
        raise ConfigError(
            f"Metric definitions file not found: {metric_file}\n"
            f"Please create metric_definitions.json with metric index mappings."
        )

    # Read and parse the metric definitions file
    try:
        with metric_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in metric_definitions.json: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Could not read metric definitions file {metric_file}: {exc}") from exc

    # Validate that the required 'metric_definitions' key is present
    if "metric_definitions" not in data:
        raise ConfigError(
            "metric_definitions.json must contain a 'metric_definitions' key"
        )

    # Extract the metric definitions dictionary from the loaded data
    metric_definitions = data["metric_definitions"]

    # Validate that the 'metric_definitions' value is itself a dictionary
    if not isinstance(metric_definitions, dict):
        raise ConfigError(
            "'metric_definitions' in metric_definitions.json must be a dictionary."
        )

    return metric_definitions