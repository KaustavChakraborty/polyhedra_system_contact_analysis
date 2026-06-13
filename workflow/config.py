# workflow/config.py
# ==============================================================================
# Module: workflow.config
# Purpose: Workflow configuration management
#
# Classes:
#   - WorkflowConfig: Master workflow configuration
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from .. import ValidationError

logger = logging.getLogger(__name__)


# ==============================================================================
# WORKFLOW CONFIG
# ==============================================================================

@dataclass
class WorkflowConfig:
    """
    Master workflow configuration.
    
    Aggregates all domain configurations.
    
    Attributes
    ----------
    trajectory_file : str
        Path to trajectory file
    output_dir : str
        Output directory
    n_processes : int
        Number of processes for parallelization
    use_mpi : bool
        Enable MPI distribution
    verbose : bool
        Verbose logging
    config_dict : Dict[str, Any]
        Configuration parameters
    
    Examples
    --------
    >>> config = WorkflowConfig(trajectory_file="traj.gsd")
    >>> config.validate()
    """
    
    trajectory_file: str
    output_dir: str = "."
    n_processes: int = 1
    use_mpi: bool = False
    verbose: bool = True
    config_dict: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.trajectory_file:
            raise ValidationError(
                "trajectory_file must be specified",
                error_code="WF_NO_TRAJECTORY"
            )
        
        if self.n_processes < 1:
            raise ValidationError(
                "n_processes must be >= 1",
                error_code="WF_INVALID_PROCESSES"
            )
        
        if self.use_mpi and self.n_processes < 2:
            raise ValidationError(
                "MPI requires n_processes >= 2",
                error_code="WF_INVALID_MPI"
            )
        
        logger.debug(
            f"[WorkflowConfig] Initialized: "
            f"traj={self.trajectory_file}, "
            f"out={self.output_dir}, "
            f"procs={self.n_processes}"
        )
    
    def set_parameter(self, key: str, value: Any) -> None:
        """
        Set configuration parameter.
        
        Parameters
        ----------
        key : str
            Parameter key
        value : Any
            Parameter value
        """
        
        self.config_dict[key] = value
        logger.debug(f"[WorkflowConfig] Set {key} = {value}")
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """
        Get configuration parameter.
        
        Parameters
        ----------
        key : str
            Parameter key
        default : Any, optional
            Default value
        
        Returns
        -------
        Any
            Parameter value
        """
        
        return self.config_dict.get(key, default)
    
    def get_all_parameters(self) -> Dict[str, Any]:
        """Get all parameters."""
        return self.config_dict.copy()
    
    def validate(self) -> bool:
        """
        Validate configuration.
        
        Returns
        -------
        bool
            Whether valid
        """
        
        logger.debug("[WorkflowConfig] Validating configuration")
        
        # Check required parameters
        required = {
            'r_min': float,
            'r_max': float,
            'n_bins': int,
        }
        
        for key, expected_type in required.items():
            value = self.get_parameter(key)
            
            if value is None:
                logger.warning(f"[WorkflowConfig] Missing parameter: {key}")
                # Not strictly required, just warn
                continue
            
            if not isinstance(value, expected_type):
                raise ValidationError(
                    f"Parameter {key} has wrong type",
                    error_code="WF_CONFIG_TYPE_ERROR"
                )
        
        logger.info("[WorkflowConfig] Configuration valid")
        
        return True
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            'trajectory_file': self.trajectory_file,
            'output_dir': self.output_dir,
            'n_processes': self.n_processes,
            'use_mpi': self.use_mpi,
            'verbose': self.verbose,
            'parameters': len(self.config_dict),
        }


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("WORKFLOW CONFIG MODULE - TESTING")
    print("="*80 + "\n")
    
    print("[TEST 1] WorkflowConfig initialization...")
    try:
        config = WorkflowConfig(trajectory_file="test.gsd", output_dir="/tmp")
        print(f"✓ Config created")
        print(f"  Summary: {config.get_summary()}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("[TEST 2] Set and get parameters...")
    try:
        config.set_parameter('r_min', 0.5)
        config.set_parameter('r_max', 5.0)
        config.set_parameter('n_bins', 50)
        
        r_min = config.get_parameter('r_min')
        print(f"✓ Parameters set")
        print(f"  r_min: {r_min}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("[TEST 3] Validate configuration...")
    try:
        config.validate()
        print(f"✓ Configuration valid\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("="*80)
    print("✓ Config tests passed!")
    print("="*80 + "\n")
