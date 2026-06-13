# workflow/__init__.py
# ==============================================================================
# Module: workflow
# Purpose: High-level orchestration domain - pipeline coordination
#
# Exports:
#   - WorkflowConfig (configuration)
#   - WorkflowState, WorkflowStage (state management)
#   - ParallelManager, MPIHelper (parallelization)
#   - AnalysisPipeline (main orchestration)
#
# Author: Contact Analysis Team
# ==============================================================================

from .config import WorkflowConfig
from .state import WorkflowState, WorkflowStage
from .parallel import ParallelManager, MPIHelper
from .pipeline import AnalysisPipeline

__all__ = [
    # Configuration
    'WorkflowConfig',
    
    # State management
    'WorkflowState',
    'WorkflowStage',
    
    # Parallelization
    'ParallelManager',
    'MPIHelper',
    
    # Pipeline
    'AnalysisPipeline',
]

__version__ = '1.0.0'
__author__ = 'Contact Analysis Team'
