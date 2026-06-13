# workflow/state.py
# ==============================================================================
# Module: workflow.state
# Purpose: Workflow state management
#
# Classes:
#   - WorkflowState: Track workflow execution state
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum

from .. import ValidationError

logger = logging.getLogger(__name__)


# ==============================================================================
# WORKFLOW STAGES
# ==============================================================================

class WorkflowStage(Enum):
    """Workflow execution stages."""
    INITIALIZED = "initialized"
    LOADING_CONFIG = "loading_config"
    LOADING_TRAJECTORY = "loading_trajectory"
    PROCESSING_PARTICLES = "processing_particles"
    DETECTING_CONTACTS = "detecting_contacts"
    COMPUTING_METRICS = "computing_metrics"
    ANALYZING_RESULTS = "analyzing_results"
    GENERATING_VISUALIZATION = "generating_visualization"
    EXPORTING_RESULTS = "exporting_results"
    COMPLETED = "completed"
    FAILED = "failed"


# ==============================================================================
# WORKFLOW STATE
# ==============================================================================

@dataclass
class WorkflowState:
    """
    Track workflow execution state.
    
    Maintains current stage, progress, and results.
    
    Attributes
    ----------
    current_stage : WorkflowStage
        Current execution stage
    progress : float
        Overall progress (0-100%)
    results : Dict[str, Any]
        Accumulated results
    errors : List[str]
        Accumulated errors
    frames_processed : int
        Number of frames processed
    
    Examples
    --------
    >>> state = WorkflowState()
    >>> state.advance_stage(WorkflowStage.LOADING_TRAJECTORY)
    >>> state.update_progress(25.0)
    """
    
    current_stage: WorkflowStage = WorkflowStage.INITIALIZED
    progress: float = 0.0
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    frames_processed: int = 0
    
    def __post_init__(self) -> None:
        """Initialize state."""
        logger.debug("[WorkflowState] Initialized")
    
    def advance_stage(self, stage: WorkflowStage) -> None:
        """
        Advance to next stage.
        
        Parameters
        ----------
        stage : WorkflowStage
            Next stage
        """
        
        old_stage = self.current_stage
        self.current_stage = stage
        
        logger.info(
            f"[WorkflowState] Advanced: {old_stage.value} → {stage.value}"
        )
    
    def update_progress(self, progress: float) -> None:
        """
        Update progress.
        
        Parameters
        ----------
        progress : float
            Progress percentage (0-100)
        """
        
        if progress < 0 or progress > 100:
            raise ValidationError(
                f"Progress must be 0-100, got {progress}",
                error_code="WF_INVALID_PROGRESS"
            )
        
        self.progress = progress
        logger.debug(f"[WorkflowState] Progress: {progress:.1f}%")
    
    def add_result(self, key: str, value: Any) -> None:
        """
        Add result.
        
        Parameters
        ----------
        key : str
            Result key
        value : Any
            Result value
        """
        
        self.results[key] = value
        logger.debug(f"[WorkflowState] Result added: {key}")
    
    def get_result(self, key: str, default: Any = None) -> Any:
        """
        Get result.
        
        Parameters
        ----------
        key : str
            Result key
        default : Any, optional
            Default value
        
        Returns
        -------
        Any
            Result value
        """
        
        return self.results.get(key, default)
    
    def get_all_results(self) -> Dict[str, Any]:
        """Get all results."""
        return self.results.copy()
    
    def add_error(self, error_message: str) -> None:
        """
        Add error.
        
        Parameters
        ----------
        error_message : str
            Error description
        """
        
        self.errors.append(error_message)
        logger.warning(f"[WorkflowState] Error: {error_message}")
    
    def get_errors(self) -> List[str]:
        """Get all errors."""
        return self.errors.copy()
    
    def has_errors(self) -> bool:
        """Check if any errors occurred."""
        return len(self.errors) > 0
    
    def increment_frames(self, count: int = 1) -> None:
        """Increment frame counter."""
        self.frames_processed += count
        logger.debug(f"[WorkflowState] Frames processed: {self.frames_processed}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get state summary."""
        return {
            'current_stage': self.current_stage.value,
            'progress': self.progress,
            'frames_processed': self.frames_processed,
            'results_count': len(self.results),
            'errors_count': len(self.errors),
            'has_errors': self.has_errors(),
        }
    
    def is_complete(self) -> bool:
        """Check if workflow completed."""
        return self.current_stage == WorkflowStage.COMPLETED
    
    def is_failed(self) -> bool:
        """Check if workflow failed."""
        return self.current_stage == WorkflowStage.FAILED


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("WORKFLOW STATE MODULE - TESTING")
    print("="*80 + "\n")
    
    print("[TEST 1] WorkflowState initialization...")
    try:
        state = WorkflowState()
        print(f"✓ State created")
        print(f"  Summary: {state.get_summary()}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("[TEST 2] Advance stages...")
    try:
        state.advance_stage(WorkflowStage.LOADING_TRAJECTORY)
        state.update_progress(25.0)
        
        state.advance_stage(WorkflowStage.DETECTING_CONTACTS)
        state.update_progress(50.0)
        
        print(f"✓ Stages advanced")
        print(f"  Current: {state.current_stage.value}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("[TEST 3] Results and errors...")
    try:
        state.add_result('rdf_computed', True)
        state.add_result('contact_count', 5000)
        
        state.add_error("Warning: some contacts invalid")
        
        print(f"✓ Results and errors added")
        print(f"  Errors: {state.has_errors()}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("="*80)
    print("✓ State tests passed!")
    print("="*80 + "\n")
