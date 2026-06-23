"""
Configuration validation for the contact-analysis workflow.

Overview
--------
This module validates the configuration parameters loaded from the JSON file
and normalizes them into a format suitable for the contact-analysis workflow.

Design Philosophy
-----------------
Validation is separated into helper functions, each responsible for validating
and normalizing a specific type of parameter:

    - require_keys() : Check that required keys are present
    - as_nonempty_string() : Convert and validate strings
    - as_positive_int() : Convert and validate positive integers
    - as_number() : Convert and validate numeric values
    - parse_distance_definition_indices() : Validate metric index lists
    - resolve_path() : Resolve and validate file/directory paths
    - parse_optional_rdf_cutoffs() : Validate optional RDF parameters
    - validate_and_build_config() : Main entry point orchestrating all validation

Each helper raises ConfigError with a descriptive message on failure,
allowing callers to handle validation errors uniformly.

Important Notes
---------------
- This module does NOT perform expensive validations (e.g., reading trajectory files)
- Booleans are explicitly rejected in numeric fields to catch common JSON errors
- Path resolution follows relative-to-analysis_path convention
- Optional parameters use sensible defaults if not provided
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Union

from .errors import ConfigError
from .loader import load_metric_definitions


def require_keys(data: Dict[str, Any], required_keys: Iterable[str]) -> None:
    """
    Ensure that all required keys are present in the JSON dictionary.

    This is the first validation step. It checks that the user has provided
    all mandatory parameters in the JSON configuration file. If any are missing,
    a helpful error message is raised immediately.

    Parameters
    ----------
    data : dict
        The parsed JSON configuration dictionary to check.

    required_keys : iterable of str
        An iterable (list, tuple, set, etc.) of key names that must be present
        in the dictionary. These are the mandatory parameters that must be
        provided by the user.

    Raises
    ------
    ConfigError
        If one or more required keys are missing, with a message listing
        all missing keys:
        "Missing required parameter(s) in JSON: key1, key2, key3"

    Example
    -------
    Check that a configuration has all required parameters:

        required = ["shape_name", "num_edges", "num_faces", "analysis_path"]
        try:
            require_keys(config, required)
        except ConfigError as exc:
            print(f"Error: {exc}")
    """

    # Collect all keys that are required but missing from the data
    missing = [key for key in required_keys if key not in data]

    # If any keys are missing, raise an error with the list of missing keys
    if missing:
        missing_text = ", ".join(missing)
        raise ConfigError(f"Missing required parameter(s) in JSON: {missing_text}")


def as_nonempty_string(data: Dict[str, Any], key: str) -> str:
    """
    Read a required non-empty string parameter.

    This helper function extracts a string parameter from the configuration,
    validates that it exists and is a non-empty string, and returns the
    trimmed value.

    Parameters
    ----------
    data : dict
        The configuration dictionary containing the parameter.

    key : str
        The parameter name (dictionary key) to extract.
        Examples: "shape_name", "analysis_path", "gsd_file"

    Returns
    -------
    str
        The parameter value after stripping leading/trailing whitespace.

    Raises
    ------
    ConfigError
        If the parameter is not a string or is an empty/whitespace-only string:
        "Parameter '{key}' must be a non-empty string."

    Example
    -------
    Extract a shape name from configuration:

        shape = as_nonempty_string(config, "shape_name")
        # If config["shape_name"] = "Cube", returns "Cube"
        # If config["shape_name"] = "  ", raises ConfigError
        # If config["shape_name"] = 123, raises ConfigError
    """

    # Get the value from the dictionary
    value = data[key]

    # Check that the value is a string and is not empty after stripping whitespace
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Parameter '{key}' must be a non-empty string.")
    
    # Return the trimmed string
    return value.strip()


def as_positive_int(data: Dict[str, Any], key: str) -> int:
    """
    Read a required positive integer parameter.

    This helper function extracts an integer parameter, validates that it is
    a positive (> 0) integer, and returns the value.

    Parameters
    ----------
    data : dict
        The configuration dictionary containing the parameter.

    key : str
        The parameter name (dictionary key) to extract.
        Examples: "num_edges", "num_faces", "num_frames"

    Returns
    -------
    int
        The validated positive integer value.

    Raises
    ------
    ConfigError
        In several cases:
        1. If the value is a boolean:
           "Parameter '{key}' must be a positive integer, not bool."
        2. If the value cannot be converted to an integer:
           "Parameter '{key}' must be a positive integer."
        3. If the value is <= 0:
           "Parameter '{key}' must be > 0, got {value_int}."
    """

    # Get the value from the dictionary
    value = data[key]

    # Explicitly check for bool before int check, since bool is a subclass of int
    # This catches JSON true/false values that shouldn't be used as integers
    if isinstance(value, bool):
        raise ConfigError(f"Parameter '{key}' must be a positive integer, not bool.")

    # Try to convert the value to an integer
    try:
        value_int = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Parameter '{key}' must be a positive integer.") from exc

    # Verify that the integer is positive (> 0, not >= 0)
    if value_int <= 0:
        raise ConfigError(f"Parameter '{key}' must be > 0, got {value_int}.")

    return value_int


def as_number(data: Dict[str, Any], key: str) -> float:
    """
    Read a required numeric parameter as float.

    This helper function extracts a numeric parameter (integer or float),
    converts it to float, and returns the value. It does NOT validate the
    range; that is the caller's responsibility.

    Parameters
    ----------
    data : dict
        The configuration dictionary containing the parameter.

    key : str
        The parameter name (dictionary key) to extract.
        Examples: "packing_fraction", "r_min", "tol_for_inv_quat_calc"

    Returns
    -------
    float
        The parameter value converted to float. Note that the returned type
        is always float, even if the input was an integer.

    Raises
    ------
    ConfigError
        In two cases:
        1. If the value is a boolean:
           "Parameter '{key}' must be numeric, not bool."
        2. If the value cannot be converted to a number:
           "Parameter '{key}' must be numeric, got {value!r}."

    Notes
    -----
    - This function does NOT validate that the result is positive, finite,
      or within any particular range. The caller must perform range checks.
    - The returned value is always float, even if the input was an integer.
    - Strings that can be parsed as numbers are accepted (e.g., "3.14").

    Example
    -------
    Extract numeric parameters:

        tol = as_number(config, "tol_for_inv_quat_calc")
        # If config["tol_for_inv_quat_calc"] = 1e-6, returns 1e-6
        # If config["tol_for_inv_quat_calc"] = "0.001", returns 0.001
        # If config["tol_for_inv_quat_calc"] = true, raises ConfigError
        # If config["tol_for_inv_quat_calc"] = "invalid", raises ConfigError
    """

    # Get the value from the dictionary
    value = data[key]

    # Explicitly check for bool before float conversion
    if isinstance(value, bool):
        raise ConfigError(f"Parameter '{key}' must be numeric, not bool.")

    # Try to convert to float
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Parameter '{key}' must be numeric, got {value!r}.") from exc


def parse_distance_definition_indices(data: Dict[str, Any]) -> Union[str, list[int]]:
    """
    Parse the distance metric selector.

    This function validates and parses the "distance_definition_indices" parameter,
    which controls which distance metrics are computed during the workflow.
    The user can either request all available metrics or a specific subset.

    Accepted Formats
    ----------------
    1. The string "all" (case-insensitive)
       Use all distance definitions available in the metric registry.

    2. A list of non-negative integers [0, 1, 2, ...]
       Use only the specified metric indices.

    Return Value Semantics
    ----------------------
    - "all" (string) means: use all metrics defined in metric_definitions.json
    - [0, 1, 2] (list) means: use only these specific metric indices

    Parameters
    ----------
    data : dict
        The configuration dictionary containing the parameter at key
        "distance_definition_indices".

    Returns
    -------
    str or list[int]
        Either:
            - The string "all" (lowercase) to use all metrics
            - A list of non-negative integers representing metric indices

    Raises
    ------
    ConfigError
        In several cases:

        1. If a string is provided but is not "all":
           "Parameter 'distance_definition_indices' may be the string 'all' or ..."

        2. If not a list or the list is empty:
           "Parameter 'distance_definition_indices' must be a non-empty list ..."

        3. If the list contains a boolean:
           "Parameter 'distance_definition_indices' contains bool value ..."

        4. If the list contains a non-integer value:
           "Parameter 'distance_definition_indices' must contain only integers ..."

        5. If the list contains a negative integer:
           "Parameter 'distance_definition_indices' cannot contain negative index ..."

    Example
    -------
    Parse metric selector:

        indices = parse_distance_definition_indices(config)

        if indices == "all":
            print("Using all available metrics")
        else:
            print(f"Using metrics: {indices}")

        # Example return values:
        # "all" -> The string "all"
        # [0, 1, 2] -> The list [0, 1, 2]
        # [5] -> The list [5]
    """

    # Extract the parameter key name (used in error messages)
    key = "distance_definition_indices"
    # Get the value from the configuration
    value = data[key]

    # Handle string input: must be "all" (case-insensitive)
    if isinstance(value, str):
        if value.lower().strip() == "all":
            # User requested all available metrics
            return "all"
        # String input that is not "all" is an error
        raise ConfigError(
            f"Parameter '{key}' may be the string 'all' or a list of integers; "
            f"got {value!r}."
        )

    # Handle list input: must be a non-empty list of non-negative integers
    if not isinstance(value, list) or len(value) == 0:
        raise ConfigError(
            f"Parameter '{key}' must be a non-empty list of integers or 'all'."
        )

    # Parse each item in the list as a non-negative integer
    parsed: list[int] = []
    for item in value:
        # Explicitly check for bool (since bool is a subclass of int)
        if isinstance(item, bool):
            raise ConfigError(f"Parameter '{key}' contains bool value {item}; expected int.")
        # Try to convert to integer
        try:
            idx = int(item)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"Parameter '{key}' must contain only integers; got {item!r}."
            ) from exc
        # Check that the index is non-negative
        if idx < 0:
            raise ConfigError(f"Parameter '{key}' cannot contain negative index {idx}.")
        # Add the validated index to the result list
        parsed.append(idx)

    return parsed


def resolve_path(base_path: Path, path_value: str) -> Path:
    """
    Resolve a path that may be absolute or relative to the analysis directory.

    This function implements the path resolution strategy:
        - If the path is absolute, return it as-is
        - If the path is relative, make it relative to base_path

    Parameters
    ----------
    base_path : pathlib.Path
        The base directory for resolving relative paths.
        Typically, this is the analysis_path from the configuration.

    path_value : str
        The path to resolve. Can be:
            - An absolute path (e.g., "/home/user/data/trajectory.gsd")
            - A relative path (e.g., "data/trajectory.gsd")
            - A path with tilde (e.g., "~/data/trajectory.gsd")

    Returns
    -------
    pathlib.Path
        The resolved path as a Path object.
        - If path_value is absolute, returns the absolute path as Path
        - If path_value is relative, returns base_path / path

    Example
    -------
    Resolve relative and absolute paths:

        analysis_dir = Path("/home/user/analysis")

        # Relative path: becomes analysis_dir / "data/trajectory.gsd"
        gsd_path = resolve_path(analysis_dir, "data/trajectory.gsd")
        # Result: /home/user/analysis/data/trajectory.gsd

        # Absolute path: returned as-is
        abs_path = resolve_path(analysis_dir, "/mnt/data/trajectory.gsd")
        # Result: /mnt/data/trajectory.gsd

        # Tilde: expanded to home directory
        home_path = resolve_path(analysis_dir, "~/trajectory.gsd")
        # Result: /home/user/trajectory.gsd
    """

    # Convert the string to a Path object
    path = Path(path_value).expanduser()

    # Check if the path is absolute
    if path.is_absolute():
        return path

    # Relative paths are made relative to the base_path
    return base_path / path


def parse_optional_rdf_cutoffs(data: Dict[str, Any]) -> Union[None, tuple[float, float, float]]:
    """
    Parse the optional non-interactive RDF cutoff triple.

    Parameter Coupling
    ------------------
    RDF cutoffs are optional. If the user doesn't provide them, the workflow
    uses default RDF parameters or skips RDF calculations. If the user provides
    one but not all three, an error is raised (incomplete specification).

    Parameters
    ----------
    data : dict
        The configuration dictionary that may contain "r_min", "r_max", "r_cut".

    Returns
    -------
    None or tuple of (float, float, float)
        - None : If none of the RDF parameters are provided
        - (r_min, r_max, r_cut) : If all three are provided
        Each value is a float parsed via as_number().

    Raises
    ------
    ConfigError
        If some but not all RDF parameters are provided:
        "RDF cutoff parameters must be provided together. Missing: ..."

    Example
    -------
    Parse RDF cutoffs:

        # Case 1: All three provided
        cutoffs = parse_optional_rdf_cutoffs(config)
        if cutoffs:
            r_min, r_max, r_cut = cutoffs
            print(f"RDF range: {r_min} to {r_max}, bin size {r_cut}")

        # Case 2: None provided
        cutoffs = parse_optional_rdf_cutoffs(config)
        if cutoffs is None:
            print("Using default RDF parameters")

        # Case 3: Partial specification (ERROR)
        # If config has "r_min" and "r_max" but not "r_cut"
        cutoffs = parse_optional_rdf_cutoffs(config)
        # Raises: ConfigError: RDF cutoff parameters must be provided together...
    """

    # List the three RDF cutoff parameter names
    keys = (
        "r_min",
        "r_max",
        "r_cut",
    )

    # Check which of the three parameters are present in the configuration
    present = [
        key in data
        for key in keys
    ]

    # Validation: if ANY are present but not ALL, raise an error
    if any(present) and not all(present):
        # Find which keys are missing
        missing = [
            key
            for key in keys
            if key not in data
        ]

        # Raise error with list of missing keys
        raise ConfigError(
            "RDF cutoff parameters must be provided together. "
            f"Missing: {', '.join(missing)}"
        )

    # If none are present, return None (RDF cutoffs not configured)
    if not any(present):
        return None

    # All three are present; parse each with as_number() and return as tuple
    return tuple(
        as_number(
            data,
            key,
        )
        for key in keys
    )


def validate_and_build_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the compact JSON configuration and build runtime parameters.

    This is the main entry point for configuration validation. It orchestrates
    all validation helpers to ensure the configuration is valid, then builds
    and returns a normalized parameter dictionary suitable for passing to the
    contact_analysis workflow.

    Validation Steps
    ----------------
    1. Check that all required keys are present
    2. Validate individual parameter types and values
    3. Validate path existence and accessibility
    4. Load and validate metric definitions
    5. Build normalized output dictionary with all parameters

    Parameters
    ----------
    data : dict
        The raw configuration dictionary loaded from the parameter JSON file.
        Should contain all required parameters and may contain optional ones.

    Returns
    -------
    dict
        Cleaned and validated parameters ready to pass into the contact_analysis
        workflow. Contains all required parameters plus computed values.

        Keys in returned dictionary:
            - shape : str
                The particle shape name (e.g., "Cube")
            - analysis_path : str
                Path to the analysis directory (absolute)
            - output_path : str
                Path to the output directory (created if doesn't exist)
            - gsd_file : str
                Path to the input trajectory file
            - shape_file : str
                Path to the shape geometry JSON file
            - num_edges : int
                Number of edges in the particle shape
            - num_faces : int
                Number of faces in the particle shape
            - tol_for_inv_quat_calc : float
                Tolerance for quaternion inversion calculations
            - indices : str or list[int]
                Either "all" or a list of metric indices to use
            - dimension_index : int
                Op dimensionality index
            - num_frames : int
                Number of frames to analyze from trajectory
            - packing_fraction : str
                Packing fraction (kept as string for reproducibility)
            - num_frames_for_rdf_averaging : int
                Number of frames to average RDF over (default: 10)
            - check_particle_overlaps : bool
                Whether to check for particle overlaps (default: True)
            - r_min, r_max, r_cut : float or None
                RDF cutoff parameters (None if not specified)
            - metric_definitions : dict
                Mapping of metric indices to names (loaded from file)

    Raises
    ------
    ConfigError
        If required parameters are missing, have invalid types, point to
        missing files/directories, or contain semantically invalid values.
        The error message describes the specific problem and expected format.

    Notes
    -----
    - This function is the main place to update when new JSON parameters are
      deliberately added to the workflow.
    - Optional parameters get sensible defaults if not provided:
      * num_frames_for_rdf_averaging defaults to 10
      * check_particle_overlaps defaults to True
    - The function modifies the output_path if it doesn't exist (creates it).
    - The function validates that input files (trajectory, shape) exist.
    - Metric indices are validated against metric_definitions.json.

    Example
    -------
    Load and validate configuration:

        try:
            raw_config = load_json_config("param_file.json")
            params = validate_and_build_config(raw_config)
            print(f"Validated {len(params)} parameters")
        except ConfigError as exc:
            print(f"Configuration error: {exc}")
            sys.exit(2)

        # Now params can be passed to the workflow
        result = run_contact_analysis(
            params=params,
            shape_geometry=shape_geometry,
            comm=comm,
            rank=rank,
            size=size,
        )

    """

    # Step 1: Define and check required keys
    # These parameters must be provided by the user in the JSON file
    required_keys = [
        "shape_name",
        "num_edges",
        "num_faces",
        "analysis_path",
        "output_path",
        "input_GSD_POS_path",
        "shape_file",
        "distance_definition_indices",
        "dimension_index",
        "num_frames",
        "packing_fraction",
        "tolerance_for_inv_quat_of_body_calc",
    ]
    require_keys(data, required_keys)

    # Step 2: Validate individual string parameters
    # Shape name: non-empty string
    shape = as_nonempty_string(data, "shape_name")

    # Step 3: Resolve and validate directory paths
    # analysis_path is the base directory; must exist
    analysis_path = Path(as_nonempty_string(data, "analysis_path")).expanduser()
    if not analysis_path.is_dir():
        raise ConfigError(f"analysis_path does not exist or is not a directory: {analysis_path}")

    # output_path and file paths are resolved relative to analysis_path
    output_path = resolve_path(analysis_path, as_nonempty_string(data, "output_path"))
    gsd_file = resolve_path(analysis_path, as_nonempty_string(data, "input_GSD_POS_path"))
    shape_file = resolve_path(analysis_path, as_nonempty_string(data, "shape_file"))

    # Step 4: Validate that input files exist
    # Trajectory file is required input
    if not gsd_file.is_file():
        raise ConfigError(f"Input GSD/POS trajectory file not found: {gsd_file}")

    # Shape geometry file is required input
    if not shape_file.is_file():
        raise ConfigError(f"Shape JSON file not found: {shape_file}")

    # Step 5: Create output directory if it doesn't exist
    # The workflow writes results here
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigError(f"Could not create output directory {output_path}: {exc}") from exc

    # Step 6: Validate shape geometry parameters (positive integers)
    num_edges = as_positive_int(data, "num_edges")
    num_faces = as_positive_int(data, "num_faces")
    num_frames = as_positive_int(data, "num_frames")

    # Step 7: Validate quaternion tolerance (must be positive float)
    tol_for_inv_quat_calc = as_number(data, "tolerance_for_inv_quat_of_body_calc")
    if tol_for_inv_quat_calc <= 0:
        raise ConfigError(
            "Parameter 'tolerance_for_inv_quat_of_body_calc' must be > 0, "
            f"got {tol_for_inv_quat_calc}."
        )

    # Step 8: Validate dimension index (integer, any value)
    dimension_index = data["dimension_index"]
    if isinstance(dimension_index, bool):
        raise ConfigError("Parameter 'dimension_index' must be numeric/integer, not bool.")

    try:
        dimension_index = int(dimension_index)
    except (TypeError, ValueError) as exc:
        raise ConfigError("Parameter 'dimension_index' must be an integer-like value.") from exc

    # Step 9: Validate distance metric indices
    # Returns either "all" (string) or a list of specific indices
    indices = parse_distance_definition_indices(data)

    # Step 10: Load metric definitions and validate requested indices
    metric_definitions = load_metric_definitions()
    if isinstance(indices, list):
        # If specific indices were requested, verify they exist in the definitions
        for idx in indices:
            if str(idx) not in metric_definitions:
                raise ConfigError(
                    f"Metric index {idx} not found in metric_definitions.json. "
                    f"Valid indices are: {', '.join(sorted(metric_definitions.keys()))}"
                )

    # Step 11: Validate packing fraction (non-empty string; kept as string for reproducibility)
    packing_fraction = str(data["packing_fraction"]).strip()

    if not packing_fraction:
        raise ConfigError("Parameter 'packing_fraction' cannot be empty.")

    # Step 12: Validate optional RDF cutoff parameters
    # Either all three (r_min, r_max, r_cut) must be provided, or none
    rdf_cutoffs = parse_optional_rdf_cutoffs(data)

    # Unpack the RDF cutoffs (if provided) or set to None
    if rdf_cutoffs is None:
        r_min = None
        r_max = None
        r_cut = None
    else:
        r_min, r_max, r_cut = rdf_cutoffs

    # Step 13: Build and return the normalized configuration dictionary
    # This is passed to the workflow for execution
    return {
        "shape": shape,
        "analysis_path": str(analysis_path),
        "output_path": str(output_path),
        "gsd_file": str(gsd_file),
        "shape_file": str(shape_file),
        "num_edges": num_edges,
        "num_faces": num_faces,
        "tol_for_inv_quat_calc": tol_for_inv_quat_calc,
        "indices": indices,
        "dimension_index": dimension_index,
        "num_frames": num_frames,
        "packing_fraction": packing_fraction,
        "num_frames_for_rdf_averaging": data.get("num_frames_for_rdf_averaging", 10),
        "check_particle_overlaps": data.get("check_particle_overlaps", True),
        "r_min": r_min,
        "r_max": r_max,
        "r_cut": r_cut,
        "metric_definitions": metric_definitions,
    }