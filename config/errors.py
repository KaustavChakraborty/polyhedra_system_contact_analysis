"""
Configuration-related exceptions.

Overview
--------
This module defines custom exception types used throughout the configuration
subsystem of the contact-analysis workflow. These exceptions are raised when
errors occur during:

    - JSON file loading and parsing
    - Parameter validation and type checking
    - Path validation and resolution
    - Metric index validation
    - Directory creation and file access

Purpose
-------
By using a custom ConfigError exception, the application can cleanly separate
configuration-layer errors from other types of exceptions (I/O errors, network
errors, computational errors, etc.). This allows:

    1. Cleaner error handling in the main entry point (main.py)
    2. Clear distinction between configuration problems and runtime problems
    3. Consistent error reporting to the user

Exception Usage Pattern
-----------------------
Throughout the configuration module, ConfigError is raised with descriptive
messages when validation fails. The main entry point catches ConfigError
exceptions and prints them in a formatted way, providing the user with
actionable feedback.

Example:
    try:
        config = load_json_config("param_file.json")
        params = validate_and_build_config(config)
    except ConfigError as exc:
        print("Configuration Error:", str(exc))
        return 2  # Exit with error code
"""


class ConfigError(Exception):
    """Raised when the parameter JSON file is missing or contains invalid values."""