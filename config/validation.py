"""
Configuration validation for the contact-analysis workflow.

This module is a direct modular extraction from the old main.py validation logic.

Important:
    This file should only validate and normalize inputs.
    It should not perform RDF calculation, trajectory reading, face pairing,
    overlap calculation, distance metrics, or Cij calculation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Union

from .errors import ConfigError
from .loader import load_metric_definitions


def require_keys(data: Dict[str, Any], required_keys: Iterable[str]) -> None:
    """
    Ensure that all required keys are present in the JSON dictionary.

    Raises
    ------
    ConfigError
        If one or more required keys are missing.
    """
    missing = [key for key in required_keys if key not in data]
    if missing:
        missing_text = ", ".join(missing)
        raise ConfigError(f"Missing required parameter(s) in JSON: {missing_text}")


def as_nonempty_string(data: Dict[str, Any], key: str) -> str:
    """Read a required non-empty string parameter."""
    value = data[key]
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Parameter '{key}' must be a non-empty string.")
    return value.strip()


def as_positive_int(data: Dict[str, Any], key: str) -> int:
    """Read a required positive integer parameter."""
    value = data[key]

    if isinstance(value, bool):
        raise ConfigError(f"Parameter '{key}' must be a positive integer, not bool.")

    try:
        value_int = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Parameter '{key}' must be a positive integer.") from exc

    if value_int <= 0:
        raise ConfigError(f"Parameter '{key}' must be > 0, got {value_int}.")

    return value_int


def as_number(data: Dict[str, Any], key: str) -> float:
    """Read a required numeric parameter as float."""
    value = data[key]

    if isinstance(value, bool):
        raise ConfigError(f"Parameter '{key}' must be numeric, not bool.")

    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Parameter '{key}' must be numeric, got {value!r}.") from exc


def parse_distance_definition_indices(data: Dict[str, Any]) -> Union[str, list[int]]:
    """
    Parse the distance metric selector.

    Accepted formats
    ----------------
    1. "all"
       Use all distance definitions available in ContactDistanceMetrics.

    2. [0, 1, 2]
       Use selected metric indices from the internal metric registry.

    Returns
    -------
    str or list[int]
        Either "all" or a list of non-negative integer indices.
    """
    key = "distance_definition_indices"
    value = data[key]

    if isinstance(value, str):
        if value.lower().strip() == "all":
            return "all"
        raise ConfigError(
            f"Parameter '{key}' may be the string 'all' or a list of integers; "
            f"got {value!r}."
        )

    if not isinstance(value, list) or len(value) == 0:
        raise ConfigError(
            f"Parameter '{key}' must be a non-empty list of integers or 'all'."
        )

    parsed: list[int] = []
    for item in value:
        if isinstance(item, bool):
            raise ConfigError(f"Parameter '{key}' contains bool value {item}; expected int.")
        try:
            idx = int(item)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"Parameter '{key}' must contain only integers; got {item!r}."
            ) from exc
        if idx < 0:
            raise ConfigError(f"Parameter '{key}' cannot contain negative index {idx}.")
        parsed.append(idx)

    return parsed


def resolve_path(base_path: Path, path_value: str) -> Path:
    """
    Resolve a path that may be absolute or relative to the analysis directory.
    """
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return base_path / path


def parse_optional_rdf_cutoffs(
    data: Dict[str, Any],
) -> Union[None, tuple[float, float, float]]:
    """
    Parse the optional non-interactive RDF cutoff triple.

    The values must either all be present or all be absent. Each configured
    value is converted with float(), matching the old interactive workflow.
    No ordering, positivity, or finiteness validation is added.
    """
    keys = (
        "r_min",
        "r_max",
        "r_cut",
    )

    present = [
        key in data
        for key in keys
    ]

    if any(present) and not all(present):
        missing = [
            key
            for key in keys
            if key not in data
        ]

        raise ConfigError(
            "RDF cutoff parameters must be provided together. "
            f"Missing: {', '.join(missing)}"
        )

    if not any(present):
        return None

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

    This function is the main place to update when new JSON parameters are
    deliberately added to the workflow.

    Returns
    -------
    dict
        Cleaned and validated parameters ready to pass into contact_analysis_class.

    Raises
    ------
    ConfigError
        If required parameters are missing, have the wrong type, or point to
        missing input files/directories.
    """
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

    shape = as_nonempty_string(data, "shape_name")

    analysis_path = Path(as_nonempty_string(data, "analysis_path")).expanduser()
    if not analysis_path.is_dir():
        raise ConfigError(f"analysis_path does not exist or is not a directory: {analysis_path}")

    output_path = resolve_path(analysis_path, as_nonempty_string(data, "output_path"))
    gsd_file = resolve_path(analysis_path, as_nonempty_string(data, "input_GSD_POS_path"))
    shape_file = resolve_path(analysis_path, as_nonempty_string(data, "shape_file"))

    if not gsd_file.is_file():
        raise ConfigError(f"Input GSD/POS trajectory file not found: {gsd_file}")

    if not shape_file.is_file():
        raise ConfigError(f"Shape JSON file not found: {shape_file}")

    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigError(f"Could not create output directory {output_path}: {exc}") from exc

    num_edges = as_positive_int(data, "num_edges")
    num_faces = as_positive_int(data, "num_faces")
    num_frames = as_positive_int(data, "num_frames")

    tol_for_inv_quat_calc = as_number(data, "tolerance_for_inv_quat_of_body_calc")
    if tol_for_inv_quat_calc <= 0:
        raise ConfigError(
            "Parameter 'tolerance_for_inv_quat_of_body_calc' must be > 0, "
            f"got {tol_for_inv_quat_calc}."
        )

    dimension_index = data["dimension_index"]
    if isinstance(dimension_index, bool):
        raise ConfigError("Parameter 'dimension_index' must be numeric/integer, not bool.")

    try:
        dimension_index = int(dimension_index)
    except (TypeError, ValueError) as exc:
        raise ConfigError("Parameter 'dimension_index' must be an integer-like value.") from exc

    indices = parse_distance_definition_indices(data)

    metric_definitions = load_metric_definitions()
    if isinstance(indices, list):
        for idx in indices:
            if str(idx) not in metric_definitions:
                raise ConfigError(
                    f"Metric index {idx} not found in metric_definitions.json. "
                    f"Valid indices are: {', '.join(sorted(metric_definitions.keys()))}"
                )

    packing_fraction = str(data["packing_fraction"]).strip()

    if not packing_fraction:
        raise ConfigError("Parameter 'packing_fraction' cannot be empty.")

    rdf_cutoffs = parse_optional_rdf_cutoffs(data)

    if rdf_cutoffs is None:
        r_min = None
        r_max = None
        r_cut = None
    else:
        r_min, r_max, r_cut = rdf_cutoffs

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