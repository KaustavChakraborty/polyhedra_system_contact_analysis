# core/exceptions.py
# ==============================================================================
# Module: core.exceptions
# Purpose: Custom exception hierarchy for contact analysis workflow
#
# This module defines a hierarchy of custom exceptions that enable:
#   1. Precise error identification (which module has the problem?)
#   2. Graceful error handling (catch specific vs. general errors)
#   3. Better debugging (stack traces point to exact error type)
#   4. User-friendly messages (convert technical errors to guidance)
#
# Exception Hierarchy:
#   ContactAnalysisError (base)
#     ├── ConfigError          (configuration/parameter issues)
#     ├── GeometryError        (geometric computation failures)
#     ├── ParticleError        (particle system issues)
#     ├── ContactError         (contact detection failures)
#     ├── MetricError          (distance metric computation)
#     ├── TrajectoryError      (trajectory file I/O)
#     ├── VisualizationError   (plotting/output)
#     ├── ValidationError      (input validation)
#     └── DataTypeError        (type mismatch/conversion)
#
# Usage Pattern:
#   try:
#       result = risky_geometry_operation()
#   except GeometryError as e:
#       logger.error(f"Geometry failed: {e.friendly_message()}")
#       # Handle gracefully
#   except ContactAnalysisError as e:
#       logger.critical(f"Unexpected error: {e}")
#       raise
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
import traceback
from typing import Optional, Any

logger = logging.getLogger(__name__)


# ==============================================================================
# BASE EXCEPTION CLASS
# ==============================================================================

class ContactAnalysisError(Exception):
    """
    Base exception for all contact analysis errors.
    
    This is the parent class for all custom exceptions in the contact analysis
    workflow. Use this to catch any contact-analysis-specific error.
    
    Attributes
    ----------
    message : str
        The error message
    error_code : Optional[str]
        Machine-readable error code for programmatic handling
    context : Optional[dict]
        Additional context information for debugging
    
    Examples
    --------
    >>> try:
    ...     some_operation()
    ... except ContactAnalysisError as e:
    ...     print(f"Error: {e}")
    ...     print(f"Code: {e.error_code}")
    ...     print(f"Context: {e.context}")
    
    Notes
    -----
    Always specify a clear, actionable error message. Include:
      - What went wrong (e.g., "Face area is negative")
      - Why it matters (e.g., "This indicates invalid face geometry")
      - What to do (e.g., "Check shape.json for correct face definitions")
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[dict] = None,
    ):
        """
        Initialize ContactAnalysisError.
        
        Parameters
        ----------
        message : str
            Human-readable error message
        error_code : Optional[str]
            Machine-readable error code (e.g., "GEOMETRY_INVALID_AREA")
        context : Optional[dict]
            Additional debugging context (e.g., {"face_id": 3, "area": -0.5})
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or f"{self.__class__.__name__}"
        self.context = context or {}
        
        logger.debug(
            f"Exception raised: {self.__class__.__name__} | "
            f"Code: {self.error_code} | Message: {message}"
        )
    
    def friendly_message(self) -> str:
        """
        Return a user-friendly error message with context.
        
        This method converts technical exception details into actionable
        guidance for users.
        
        Returns
        -------
        str
            Formatted error message with context information
            
        Examples
        --------
        >>> try:
        ...     raise GeometryError("Invalid vertices", context={"n": 2})
        ... except ContactAnalysisError as e:
        ...     print(e.friendly_message())
            
            GeometryError: Invalid vertices
            Code: GEOMETRY_INVALID_VERTICES
            Context: {'n': 2}
        """
        msg = f"{self.__class__.__name__}: {self.message}\n"
        msg += f"Code: {self.error_code}"
        
        if self.context:
            msg += f"\nContext: {self.context}"
        
        return msg
    
    def __repr__(self) -> str:
        """Return detailed representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"error_code={self.error_code!r})"
        )
    
    def __str__(self) -> str:
        """Return string representation."""
        return self.message


# ==============================================================================
# DOMAIN-SPECIFIC EXCEPTION CLASSES
# ==============================================================================

class ConfigError(ContactAnalysisError):
    """
    Raised when configuration or parameter validation fails.
    
    This exception indicates problems with:
    - param_file.json loading or parsing
    - Missing required parameters
    - Parameter values out of valid range
    - Invalid parameter types
    
    Examples
    --------
    >>> raise ConfigError(
    ...     "num_frames must be positive integer",
    ...     error_code="CONFIG_INVALID_FRAMES",
    ...     context={"parameter": "num_frames", "value": -5}
    ... )
    
    Notes
    -----
    When raising ConfigError, include the parameter name and invalid value
    in the context for debugging.
    """
    pass


class GeometryError(ContactAnalysisError):
    """
    Raised when geometric computation fails.
    
    This exception indicates problems with:
    - Polygon area/normal/centroid computation
    - Convex hull or decomposition failures
    - Polygon reordering (non-coplanar vertices)
    - Invalid face definitions
    - Numerical instability in geometric calculations
    
    Examples
    --------
    >>> raise GeometryError(
    ...     "Polygon vertices are not coplanar",
    ...     error_code="GEOMETRY_NOT_COPLANAR",
    ...     context={"face_id": 3, "tolerance": 1e-12, "max_deviation": 1e-11}
    ... )
    
    Notes
    -----
    Include the face_id or shape_id and the problematic tolerance values
    to help users adjust POLYGON_REORDER_TOL if needed.
    """
    pass


class ParticleError(ContactAnalysisError):
    """
    Raised when particle system operations fail.
    
    This exception indicates problems with:
    - Particle creation or initialization
    - Particle geometry loading (invalid shape.json)
    - Particle system neighbor finding
    - AABB computation failures
    - Invalid orientation (quaternion)
    
    Examples
    --------
    >>> raise ParticleError(
    ...     "Invalid quaternion for particle orientation",
    ...     error_code="PARTICLE_INVALID_QUATERNION",
    ...     context={"particle_id": 42, "quaternion": [1, 0, 0, 0]}
    ... )
    
    Notes
    -----
    Include particle_id to help users identify problematic particles
    in their trajectory.
    """
    pass


class ContactError(ContactAnalysisError):
    """
    Raised when contact detection fails.
    
    This exception indicates problems with:
    - Overlap detection between particles
    - Face pair processing failures
    - Contact classification errors
    - Neighbor list construction
    
    Examples
    --------
    >>> raise ContactError(
    ...     "Overlap detection failed for particle pair",
    ...     error_code="CONTACT_OVERLAP_FAILED",
    ...     context={"particle_A": 10, "particle_B": 25, "reason": "degenerate polygon"}
    ... )
    
    Notes
    -----
    Include both particle IDs to help users identify problematic pairs.
    """
    pass


class MetricError(ContactAnalysisError):
    """
    Raised when distance metric computation fails.
    
    This exception indicates problems with:
    - Metric not found or not registered
    - Metric computation returns invalid result
    - Missing required inputs for metric
    - Numerical instability in metric calculation
    
    Examples
    --------
    >>> raise MetricError(
    ...     "Metric computation returned NaN",
    ...     error_code="METRIC_NAN_RESULT",
    ...     context={"metric": "vertex_to_edge_perp", "facet_A": 5, "facet_B": 12}
    ... )
    
    Notes
    -----
    Include the metric name and face indices to help debug specific metrics.
    """
    pass


class TrajectoryError(ContactAnalysisError):
    """
    Raised when trajectory file I/O fails.
    
    This exception indicates problems with:
    - GSD file not found or corrupted
    - Frame index out of range
    - Invalid frame data format
    - Box dimensions invalid
    - Position/orientation data missing
    
    Examples
    --------
    >>> raise TrajectoryError(
    ...     "Frame index out of range",
    ...     error_code="TRAJECTORY_FRAME_OUTOFRANGE",
    ...     context={"requested_frame": 150, "total_frames": 100}
    ... )
    
    Notes
    -----
    Include requested_frame and total_frames to help users understand
    the valid frame range.
    """
    pass


class VisualizationError(ContactAnalysisError):
    """
    Raised when plotting or visualization fails.
    
    This exception indicates problems with:
    - Plot generation failure
    - Output file writing error
    - Invalid plot data
    - Missing matplotlib or plotting backend
    
    Examples
    --------
    >>> raise VisualizationError(
    ...     "Cannot save plot to file",
    ...     error_code="VIZ_FILE_WRITE_ERROR",
    ...     context={"output_path": "/nonexistent/dir/plot.png"}
    ... )
    
    Notes
    -----
    Visualization errors should not stop the workflow. Catch and log,
    but continue processing if possible.
    """
    pass


class ValidationError(ContactAnalysisError):
    """
    Raised when input validation fails.
    
    This exception indicates problems with:
    - Invalid array shape or dimensions
    - Type mismatches (e.g., expected float, got string)
    - Out-of-range values
    - Missing required fields
    - Inconsistent data structures
    
    Examples
    --------
    >>> raise ValidationError(
    ...     "Vertex array has wrong shape",
    ...     error_code="VALIDATION_SHAPE_MISMATCH",
    ...     context={"expected_shape": "(N, 3)", "actual_shape": "(N, 4)"}
    ... )
    
    Notes
    -----
    ValidationError is raised early in the pipeline to catch bad inputs
    before expensive computations.
    """
    pass


class DataTypeError(ContactAnalysisError):
    """
    Raised when data type conversion or mismatch occurs.
    
    This exception indicates problems with:
    - Cannot convert value to required type
    - Type mismatch between expected and actual
    - Lost precision in type conversion
    - Incompatible data structures
    
    Examples
    --------
    >>> raise DataTypeError(
    ...     "Cannot convert value to float",
    ...     error_code="DATATYPE_CONVERSION_FAILED",
    ...     context={"value": "abc", "target_type": "float"}
    ... )
    
    Notes
    -----
    DataTypeError should include the problematic value and target type
    to help users understand what went wrong.
    """
    pass


# ==============================================================================
# EXCEPTION UTILITY FUNCTIONS
# ==============================================================================

def get_exception_chain(exc: Exception) -> list:
    """
    Extract the full exception chain for debugging.
    
    Returns a list of all exceptions in the chain, from the original
    exception up to the current one. Useful for understanding how
    an error propagated through the code.
    
    Parameters
    ----------
    exc : Exception
        The exception to analyze
    
    Returns
    -------
    list
        List of (exception_type, message) tuples in chain order
    
    Examples
    --------
    >>> try:
    ...     risky_operation()
    ... except Exception as e:
    ...     chain = get_exception_chain(e)
    ...     print("Exception chain:")
    ...     for exc_type, msg in chain:
    ...         print(f"  {exc_type.__name__}: {msg}")
    """
    chain = []
    current = exc
    
    while current is not None:
        chain.append((type(current), str(current)))
        current = current.__cause__ if hasattr(current, '__cause__') else None
    
    return chain


def format_exception_for_logging(exc: Exception, include_traceback: bool = True) -> str:
    """
    Format an exception with full context for logging.
    
    Provides a formatted exception message suitable for logging, including
    the exception hierarchy, context information (if available), and
    optionally the full traceback.
    
    Parameters
    ----------
    exc : Exception
        The exception to format
    include_traceback : bool, optional
        If True, include full traceback. Default is True.
    
    Returns
    -------
    str
        Formatted exception string
    
    Examples
    --------
    >>> try:
    ...     risky_operation()
    ... except Exception as e:
    ...     logger.error(format_exception_for_logging(e))
    """
    lines = []
    
    # Add exception type and message
    lines.append(f"\n{'='*80}")
    lines.append(f"EXCEPTION: {type(exc).__name__}")
    lines.append(f"{'='*80}")
    lines.append(f"Message: {str(exc)}")
    
    # Add context if available (for ContactAnalysisError)
    if hasattr(exc, 'context') and exc.context:
        lines.append(f"\nContext Information:")
        for key, value in exc.context.items():
            lines.append(f"  {key}: {value}")
    
    if hasattr(exc, 'error_code'):
        lines.append(f"\nError Code: {exc.error_code}")
    
    # Add exception chain
    chain = get_exception_chain(exc)
    if len(chain) > 1:
        lines.append(f"\nException Chain ({len(chain)} exceptions):")
        for i, (exc_type, msg) in enumerate(chain):
            lines.append(f"  {i+1}. {exc_type.__name__}: {msg}")
    
    # Add traceback if requested
    if include_traceback:
        lines.append(f"\nTraceback:")
        lines.append(traceback.format_exc())
    
    lines.append(f"{'='*80}\n")
    
    return '\n'.join(lines)


def safe_exception_handler(
    func,
    exception_type: type = ContactAnalysisError,
    logger_func=logger.error,
) -> Any:
    """
    Decorator for safe exception handling with logging.
    
    Wraps a function to catch exceptions, log them with full context,
    and optionally re-raise or return None.
    
    Parameters
    ----------
    func : callable
        The function to wrap
    exception_type : type, optional
        The exception type to catch. Default is ContactAnalysisError.
    logger_func : callable, optional
        The logger function to use (e.g., logger.error). Default is logger.error.
    
    Returns
    -------
    callable
        Wrapped function that catches and logs exceptions
    
    Examples
    --------
    >>> @safe_exception_handler
    ... def risky_operation():
    ...     # Some code that might raise exception
    ...     return result
    
    >>> result = risky_operation()  # Exception is caught and logged
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except exception_type as e:
            logger_func(format_exception_for_logging(e))
            raise
    
    return wrapper


if __name__ == "__main__":
    """
    Test/demo when module is run directly.
    """
    import sys
    
    print("\n" + "="*80)
    print("CORE EXCEPTIONS MODULE - TESTING")
    print("="*80 + "\n")
    
    # Test exception raising and formatting
    print("Testing exception types and formatting:\n")
    
    try:
        raise GeometryError(
            "Polygon vertices are not coplanar",
            error_code="GEOMETRY_NOT_COPLANAR",
            context={
                "face_id": 3,
                "tolerance": 1e-12,
                "max_deviation": 1e-11
            }
        )
    except GeometryError as e:
        print("Caught GeometryError:")
        print(e.friendly_message())
        print()
    
    try:
        raise MetricError(
            "Metric computation failed",
            error_code="METRIC_COMPUTATION_ERROR",
            context={"metric": "vertex_to_edge_perp", "reason": "degenerate edge"}
        )
    except MetricError as e:
        print("Caught MetricError:")
        print(e.friendly_message())
        print()
    
    try:
        raise ContactAnalysisError(
            "Unexpected error",
            error_code="UNKNOWN",
            context={"module": "workflow", "function": "run"}
        )
    except ContactAnalysisError as e:
        print("Caught base ContactAnalysisError:")
        print(e.friendly_message())
    
    print("\n" + "="*80)
    print("✓ Exceptions module is ready for use!")
    print("="*80 + "\n")