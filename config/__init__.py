"""
Configuration module for the contact-analysis workflow.

Overview
--------
This package provides a modular configuration subsystem that handles:

    1. Loading configuration from JSON files
    2. Validating configuration parameters
    3. Resolving file paths
    4. Building normalized parameter dictionaries
    5. Providing clear error messages for invalid configurations

Design Philosophy
-----------------
The configuration subsystem is designed with separation of concerns in mind:

    - errors.py : Custom exception type (ConfigError)
    - loader.py : JSON file loading and parsing
    - validation.py : Parameter validation and normalization
    - __init__.py : Public API and module-level imports

This package exposes the same configuration helpers that were previously
defined directly inside main.py, but now organized into a clean, reusable
module structure.

Module Organization
-------------------
The configuration package is broken into logical parts:

    errors.ConfigError
        The custom exception type raised for all configuration errors.
        Allows clean error handling at the application level.

    loader.load_json_config(path)
        Loads and parses the parameter JSON file.
        Handles file I/O errors and JSON syntax errors.

    loader.load_metric_definitions(path)
        Loads the metric definitions mapping file.
        Used to validate and describe distance metrics.

    validation.require_keys(data, keys)
        Checks that all required parameters are present.

    validation.as_nonempty_string(data, key)
        Validates and extracts a string parameter.

    validation.as_positive_int(data, key)
        Validates and extracts a positive integer parameter.

    validation.as_number(data, key)
        Validates and extracts a numeric parameter.

    validation.parse_distance_definition_indices(data)
        Parses the metric index selector ("all" or [0, 1, 2, ...]).

    validation.resolve_path(base, path_str)
        Resolves absolute and relative paths.

    validation.validate_and_build_config(data)
        Main entry point: validates entire configuration and builds
        normalized parameter dictionary.

Usage Pattern
-------------
Typical usage in the main entry point:

    from config import load_json_config, validate_and_build_config, ConfigError

    try:
        # Step 1: Load the raw JSON configuration
        raw_config = load_json_config("param_file.json")

        # Step 2: Validate and normalize the configuration
        params = validate_and_build_config(raw_config)

        # Step 3: Use the validated parameters
        result = run_contact_analysis(params=params, ...)

    except ConfigError as exc:
        # Handle configuration error
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(2)

Error Handling
--------------
All configuration operations raise ConfigError on failure. This exception
should be caught at the application's entry point (main.py) and handled
with a formatted error message displayed to the user before exit.

The exception message is always descriptive and actionable, explaining:
    - What parameter or file caused the problem
    - What went wrong (missing, wrong type, out of range, etc.)
    - (When helpful) what the expected format is

Example Error Messages:
    - "Missing required parameter(s) in JSON: shape_name, analysis_path"
    - "Parameter 'num_edges' must be a positive integer, got 0."
    - "Input GSD/POS trajectory file not found: /path/to/trajectory.gsd"
    - "RDF cutoff parameters must be provided together. Missing: r_min, r_cut"

Input Files
-----------
The configuration subsystem expects two files:

    1. param_file.json (user-specified, default "param_file.json")
       Contains all runtime parameters for the workflow.
       Example keys: shape_name, num_edges, analysis_path, etc.

    2. metric_definitions.json (typically in current working directory)
       Contains human-readable names for distance metrics.
       Maps metric indices (0, 1, 2, ...) to descriptive names.

Output
------
The main output of this module is the validated and normalized parameter
dictionary returned by validate_and_build_config(). This dictionary is passed
directly to the contact-analysis workflow and contains:

    - All required parameters with validated types and ranges
    - Resolved absolute paths (files and directories)
    - Optional parameters with sensible defaults applied
    - Loaded metric definitions for reference
"""

from .errors import ConfigError
from .loader import load_json_config, load_metric_definitions
from .validation import (
    require_keys,
    as_nonempty_string,
    as_positive_int,
    as_number,
    parse_distance_definition_indices,
    resolve_path,
    validate_and_build_config,
)

# Only items listed here should be considered part of the stable interface
__all__ = [
    # Exception type
    "ConfigError",
    # Loading functions
    "load_json_config",
    "load_metric_definitions",
    # Validation helper functions (for advanced use)
    "require_keys",
    "as_nonempty_string",
    "as_positive_int",
    "as_number",
    "parse_distance_definition_indices",
    "resolve_path",
    # Main entry point
    "validate_and_build_config",
]