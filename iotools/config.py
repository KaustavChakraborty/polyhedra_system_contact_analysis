# iotools/config.py
# ==============================================================================
# PURPOSE OF THIS MODULE
# ----------------------
# This file is responsible for one specific job in the modularized contact-
# analysis project:
#
#     read a JSON configuration file from disk,
#     store the loaded values in memory,
#     provide convenient access to those values,
#     optionally validate the loaded values against an expected schema,
#     and optionally save the configuration back to disk.
#
# In my current control flow, main.py calls this module very early:
#
#     main.py
#       └── ConfigLoader().load("configs/config_default.json")
#             └── json.load(...)
#
# After this file returns a Python dictionary, main.py extracts pieces such as:
#
#     config_dict["trajectory"]["file"]
#     config_dict["output"]["results_dir"]
#
# IMPORTANT DESIGN IDEA
# ---------------------
# The analysis pipeline should not directly open JSON files everywhere. Instead,
# one small module should be responsible for loading and checking configuration.
# That makes the rest of the project cleaner.
#
#
# Author: Kaustav Chakraborty
# ==============================================================================

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Mapping, Type

# from .. import ValidationError, DataTypeError
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from core import ValidationError, DataTypeError

# Module-level logger. It does not configure logging by itself; main.py does that.
logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIG LOADER
# ==============================================================================

@dataclass
class ConfigLoader:
    """
    Load and validate configuration files.
    
    Loads JSON configuration files and validates parameters.
    
    Attributes
    ----------
    config_path : Path
        Path to configuration file
    config : Dict[str, Any]
        Loaded configuration
    
    Examples
    --------
    >>> loader = ConfigLoader("param_file.json")
    >>> config = loader.load()
    """
    
    config_path: Optional[Path] = None
    config: Dict[str, Any] = None
    
    def __post_init__(self) -> None:
        """Initialize loader. Here we convert a string path into a ``Path`` object so that the rest of
        the class can use a consistent path interface."""
        if self.config_path is None:
            self.config = {}
        
        logger.debug("[ConfigLoader] Initialized")
    
    # --------------------------------------------------------------------------
    # Main entry point used by main.py
    # --------------------------------------------------------------------------
    def load(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Control flow
        ------------
        1. If the caller provides ``config_path``, update ``self.config_path``.
        2. Check that a path is available.
        3. Check that the path points to an actual file.
        4. Open the file.
        5. Parse JSON into a Python object using ``json.load``.
        6. Confirm that the top-level object is a dictionary.
        7. Store the dictionary in ``self.config``.
        8. Return that dictionary to the caller.

        Parameters
        ----------
        config_path : str, optional
            Path to configuration file
        
        Returns
        -------
        Dict[str, Any]
            Loaded configuration
        
        Raises
        ------
        ValidationError
            If file not found or invalid JSON
        
        Examples
        --------
        >>> loader = ConfigLoader()
        >>> config = loader.load("param_file.json")
        """
        
        # Allow either construction-time path or load-time path.
        if config_path:
            self.config_path = Path(config_path)
        
        # No path was given at all.
        if self.config_path is None:
            raise ValidationError(
                "No configuration path provided",
                error_code="CONFIG_NO_PATH"
            )

        path = Path(self.config_path).expanduser()

        # ``exists`` would return True for directories also, but we need a file.
        if not path.is_file():
            raise ValidationError(
                f"Configuration file not found: {path}",
                error_code="CONFIG_FILE_NOT_FOUND",
            )
        
        logger.debug(f"[ConfigLoader] Loading config from {self.config_path}")
        
        try:
            with path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)

        except json.JSONDecodeError as exc:
            # This means the file exists, but the JSON syntax is malformed.
            # Example causes: missing comma, extra trailing comma, unmatched brace.
            raise ValidationError(
                f"Invalid JSON in configuration file {path}: "
                f"line {exc.lineno}, column {exc.colno}: {exc.msg}",
                error_code="CONFIG_INVALID_JSON",
            ) from exc

        except OSError as exc:
            # This catches file-system level errors, for example permission denied.
            raise ValidationError(
                f"Could not read configuration file {path}: {exc}",
                error_code="CONFIG_LOAD_ERROR",
            ) from exc

        except Exception as exc:
            # Last-resort wrapper so the caller sees a project-specific exception.
            raise ValidationError(
                f"Failed to load configuration from {path}: {exc}",
                error_code="CONFIG_LOAD_ERROR",
            ) from exc

        # For your project, the config should be a dictionary like:
        #
        #     {
        #       "trajectory": {...},
        #       "output": {...},
        #       "analysis": {...}
        #     }
        #
        # A top-level JSON list would parse correctly, but main.py would fail later
        # because it expects dictionary methods such as config_dict.get(...).
        if not isinstance(loaded, dict):
            raise ValidationError(
                f"Configuration root must be a JSON object/dictionary, "
                f"got {type(loaded).__name__}",
                error_code="CONFIG_ROOT_NOT_DICT",
            )

        self.config_path = path
        self.config = loaded

        logger.info("[ConfigLoader] Configuration loaded from %s with %d top-level keys", path, len(self.config))

        return self.config
    
    # --------------------------------------------------------------------------
    # Convenience access method
    # --------------------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        """
        Return one top-level configuration value.

        This is just a thin wrapper around ``dict.get``:

            self.config.get(key, default)

        It is useful for flat keys. For nested values, use normal dictionary
        access or chained ``get`` calls in the calling code:

            trajectory_file = config.get("trajectory", {}).get("file")

        Parameters
        ----------
        key : str
            Top-level configuration key.

        default : Any
            Value to return if the key does not exist.

        Returns
        -------
        Any
            The stored value, or ``default`` if absent.
        """
        
        return self.config.get(key, default)

    # --------------------------------------------------------------------------
    # Simple schema validation
    # --------------------------------------------------------------------------
    def validate(self, schema: Dict[str, type] = None) -> bool:
        """
        Validate configuration against schema.

        Expected schema format
        ----------------------
        The schema is a dictionary mapping key names to expected Python types:

            schema = {
                "trajectory": dict,
                "output": dict,
                "analysis": dict,
            }

        Then this method checks:

            1. Is each required key present?
            2. Does each key have the expected type?
        
        Parameters
        ----------
        schema : Dict[str, type], optional
            Expected types: {key: type}
        
        Returns
        -------
        bool
            Whether configuration is valid
        
        Raises
        ------
        ValidationError
            If validation fails
        """
        
        if schema is None:
            logger.debug("[ConfigLoader] No schema supplied; skipping validation")
            return True
        
        logger.debug("[ConfigLoader] Validating configuration against schema")
        
        for key, expected_type in schema.items():
            if key not in self.config:
                raise ValidationError(
                    f"Required key missing: {key}",
                    error_code="CONFIG_MISSING_KEY"
                )
            
            value = self.config[key]
            
            if not isinstance(value, expected_type):
                raise DataTypeError(
                    f"Key '{key}' has type {type(value).__name__}, "
                    f"expected {expected_type.__name__}",
                    error_code="CONFIG_TYPE_ERROR"
                )
        
        logger.info("[ConfigLoader] Configuration validated successfully")
        
        return True
    
    # --------------------------------------------------------------------------
    # Save current in-memory config
    # --------------------------------------------------------------------------
    def save(self, output_path: str) -> None:
        """
        Save configuration to file.
        
        Parameters
        ----------
        output_path : str
            Output file path
        """
        
        path = Path(output_path).expanduser()

        # Make parent directories automatically if they do not exist.
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        logger.debug("[ConfigLoader] Saving configuration to %s", path)

        try:
            with path.open("w", encoding="utf-8") as handle:
                json.dump(self.config, handle, indent=2, sort_keys=False)
                handle.write("\n")

        except OSError as exc:
            raise ValidationError(
                f"Could not save configuration to {path}: {exc}",
                error_code="CONFIG_SAVE_ERROR",
            ) from exc

        logger.info("[ConfigLoader] Configuration saved to %s", path)


# ==============================================================================
# DIRECT MODULE TEST
# ==============================================================================
#
# This block runs only when the file is executed directly:
#
#     python3.8 iotools/config.py
#
# It does not run when main.py imports ConfigLoader.
# ============================================================================== 

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("IOTOOLS CONFIG MODULE - BASIC SELF TEST")
    print("=" * 80 + "\n")

    print("[TEST 1] ConfigLoader initialization")
    loader = ConfigLoader()
    print("[SUCCESS] Loader created")

    print("\n[TEST 2] Manual in-memory configuration")
    loader.config = {
        "tolerance": 1e-12,
        "n_bins": 50,
        "metrics": ["distance", "area"],
    }
    print(f"[SUCCESS] tolerance = {loader.get('tolerance')}")
    print(f"[SUCCESS] n_bins    = {loader.get('n_bins')}")

    print("\n[TEST 3] Simple schema validation")
    schema = {
        "tolerance": float,
        "n_bins": int,
        "metrics": list,
    }
    loader.validate(schema)
    print("[SUCCESS] Configuration validated")

    print("\n" + "=" * 80)
    print("[SUCCESS] Config self-test passed")
    print("=" * 80 + "\n")
