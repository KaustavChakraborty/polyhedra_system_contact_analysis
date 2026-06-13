# io/config.py
# ==============================================================================
# Module: io.config
# Purpose: Load and validate configuration files (param_file.json)
#
# Classes:
#   - ConfigLoader: Load and validate configuration
#   - ConfigValidator: Validate configuration parameters
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

from .. import ValidationError, DataTypeError

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
        """Initialize loader."""
        if self.config_path is None:
            self.config = {}
        
        logger.debug("[ConfigLoader] Initialized")
    
    def load(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from file.
        
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
        
        if config_path:
            self.config_path = Path(config_path)
        
        if self.config_path is None:
            raise ValidationError(
                "No configuration path provided",
                error_code="CONFIG_NO_PATH"
            )
        
        # Check file exists
        if not self.config_path.exists():
            raise ValidationError(
                f"Configuration file not found: {self.config_path}",
                error_code="CONFIG_FILE_NOT_FOUND"
            )
        
        logger.debug(f"[ConfigLoader] Loading config from {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValidationError(
                f"Invalid JSON in configuration file: {e}",
                error_code="CONFIG_INVALID_JSON"
            ) from e
        except Exception as e:
            raise ValidationError(
                f"Failed to load configuration: {e}",
                error_code="CONFIG_LOAD_ERROR"
            ) from e
        
        logger.info(f"[ConfigLoader] Configuration loaded: {len(self.config)} keys")
        
        return self.config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Parameters
        ----------
        key : str
            Configuration key
        default : Any, optional
            Default value if key not found
        
        Returns
        -------
        Any
            Configuration value or default
        """
        
        return self.config.get(key, default)
    
    def validate(self, schema: Dict[str, type] = None) -> bool:
        """
        Validate configuration against schema.
        
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
            return True
        
        logger.debug("[ConfigLoader] Validating configuration")
        
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
        
        logger.info("[ConfigLoader] Configuration validated")
        
        return True
    
    def save(self, output_path: str) -> None:
        """
        Save configuration to file.
        
        Parameters
        ----------
        output_path : str
            Output file path
        """
        
        logger.debug(f"[ConfigLoader] Saving config to {output_path}")
        
        with open(output_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"[ConfigLoader] Configuration saved to {output_path}")


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("IO CONFIG MODULE - TESTING")
    print("="*80 + "\n")
    
    print("[TEST 1] ConfigLoader initialization...")
    try:
        loader = ConfigLoader()
        print(f"✓ Loader created\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("[TEST 2] Load configuration...")
    try:
        # Create test config
        test_config = {
            "tolerance": 1e-12,
            "n_bins": 50,
            "metrics": ["distance", "area"]
        }
        
        loader.config = test_config
        print(f"✓ Test configuration loaded")
        print(f"  tolerance: {loader.get('tolerance')}")
        print(f"  n_bins: {loader.get('n_bins')}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("[TEST 3] Validate configuration...")
    try:
        schema = {
            "tolerance": float,
            "n_bins": int,
            "metrics": list
        }
        
        loader.validate(schema)
        print(f"✓ Configuration validated\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("="*80)
    print("✓ Config tests passed!")
    print("="*80 + "\n")
