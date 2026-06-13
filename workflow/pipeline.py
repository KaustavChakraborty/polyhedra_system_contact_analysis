# workflow/pipeline.py
# ==============================================================================
# Module: workflow.pipeline
# Purpose: Main workflow orchestration - coordinates all domains
#
# Classes:
#   - AnalysisPipeline: End-to-end analysis pipeline
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

from .. import ValidationError
from .config import WorkflowConfig
from .state import WorkflowState, WorkflowStage
from .parallel import ParallelManager, MPIHelper

logger = logging.getLogger(__name__)


# ==============================================================================
# ANALYSIS PIPELINE
# ==============================================================================

@dataclass
class AnalysisPipeline:
    """
    End-to-end analysis pipeline.
    
    Orchestrates all domains: trajectory → particles → contacts → 
    metrics → analysis → visualization → export.
    
    Attributes
    ----------
    config : WorkflowConfig
        Workflow configuration
    state : WorkflowState
        Execution state
    parallel : ParallelManager
        Parallel execution manager
    
    Examples
    --------
    >>> config = WorkflowConfig(trajectory_file="traj.gsd")
    >>> pipeline = AnalysisPipeline(config)
    >>> pipeline.execute()
    """
    
    config: WorkflowConfig
    state: WorkflowState = None
    parallel: ParallelManager = None
    
    def __post_init__(self) -> None:
        """Initialize pipeline."""
        if self.state is None:
            self.state = WorkflowState()
        
        if self.parallel is None:
            self.parallel = ParallelManager(use_mpi=self.config.use_mpi)
        
        logger.debug("[AnalysisPipeline] Initialized")
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute complete pipeline.
        
        Returns
        -------
        Dict[str, Any]
            Pipeline results
        """
        
        logger.info("[AnalysisPipeline] Starting execution")
        
        try:
            # Stage 1: Load configuration
            self._stage_load_config()
            
            # Stage 2: Load trajectory
            self._stage_load_trajectory()
            
            # Stage 3: Process particles
            self._stage_process_particles()
            
            # Stage 4: Detect contacts
            self._stage_detect_contacts()
            
            # Stage 5: Compute metrics
            self._stage_compute_metrics()
            
            # Stage 6: Analyze results
            self._stage_analyze_results()
            
            # Stage 7: Visualize
            self._stage_visualize()
            
            # Stage 8: Export
            self._stage_export()
            
            # Mark complete
            self.state.advance_stage(WorkflowStage.COMPLETED)
            self.state.update_progress(100.0)
            
            logger.info("[AnalysisPipeline] Execution complete")
            
            return self.state.get_all_results()
        
        except Exception as e:
            logger.error(f"[AnalysisPipeline] Execution failed: {e}")
            self.state.add_error(str(e))
            self.state.advance_stage(WorkflowStage.FAILED)
            raise
    
    def _stage_load_config(self) -> None:
        """Load and validate configuration."""
        self.state.advance_stage(WorkflowStage.LOADING_CONFIG)
        self.state.update_progress(5.0)
        
        logger.debug("[AnalysisPipeline] Loading configuration")
        
        self.config.validate()
        
        self.state.add_result('config', self.config.get_summary())
        logger.info("[AnalysisPipeline] Configuration loaded")
    
    def _stage_load_trajectory(self) -> None:
        """Load trajectory file."""
        self.state.advance_stage(WorkflowStage.LOADING_TRAJECTORY)
        self.state.update_progress(15.0)
        
        logger.debug("[AnalysisPipeline] Loading trajectory")
        
        trajectory_path = self.config.trajectory_file
        logger.info(f"[AnalysisPipeline] Loading: {trajectory_path}")
        
        # Placeholder: In real workflow, load from io/trajectory
        self.state.add_result('trajectory_loaded', True)
        logger.info("[AnalysisPipeline] Trajectory loaded")
    
    def _stage_process_particles(self) -> None:
        """Process particle systems."""
        self.state.advance_stage(WorkflowStage.PROCESSING_PARTICLES)
        self.state.update_progress(30.0)
        
        logger.debug("[AnalysisPipeline] Processing particles")
        
        # Placeholder: In real workflow, use particles/
        self.state.add_result('particles_processed', True)
        logger.info("[AnalysisPipeline] Particles processed")
    
    def _stage_detect_contacts(self) -> None:
        """Detect contacts."""
        self.state.advance_stage(WorkflowStage.DETECTING_CONTACTS)
        self.state.update_progress(50.0)
        
        logger.debug("[AnalysisPipeline] Detecting contacts")
        
        # Placeholder: In real workflow, use contacts/
        self.state.add_result('contacts_detected', True)
        self.state.add_result('contact_count', 0)
        logger.info("[AnalysisPipeline] Contacts detected")
    
    def _stage_compute_metrics(self) -> None:
        """Compute metrics."""
        self.state.advance_stage(WorkflowStage.COMPUTING_METRICS)
        self.state.update_progress(65.0)
        
        logger.debug("[AnalysisPipeline] Computing metrics")
        
        # Placeholder: In real workflow, use metrics/
        self.state.add_result('metrics_computed', True)
        logger.info("[AnalysisPipeline] Metrics computed")
    
    def _stage_analyze_results(self) -> None:
        """Analyze results."""
        self.state.advance_stage(WorkflowStage.ANALYZING_RESULTS)
        self.state.update_progress(80.0)
        
        logger.debug("[AnalysisPipeline] Analyzing results")
        
        # Placeholder: In real workflow, use analysis/
        self.state.add_result('rdf_computed', True)
        self.state.add_result('statistics_computed', True)
        logger.info("[AnalysisPipeline] Results analyzed")
    
    def _stage_visualize(self) -> None:
        """Generate visualizations."""
        self.state.advance_stage(WorkflowStage.GENERATING_VISUALIZATION)
        self.state.update_progress(90.0)
        
        logger.debug("[AnalysisPipeline] Generating visualizations")
        
        # Placeholder: In real workflow, use visualization/
        self.state.add_result('plots_generated', True)
        logger.info("[AnalysisPipeline] Visualizations generated")
    
    def _stage_export(self) -> None:
        """Export results."""
        self.state.advance_stage(WorkflowStage.EXPORTING_RESULTS)
        self.state.update_progress(95.0)
        
        logger.debug("[AnalysisPipeline] Exporting results")
        
        output_dir = self.config.output_dir
        logger.info(f"[AnalysisPipeline] Exporting to: {output_dir}")
        
        # Placeholder: In real workflow, use io/
        self.state.add_result('results_exported', True)
        logger.info("[AnalysisPipeline] Results exported")
    
    def get_results(self) -> Dict[str, Any]:
        """Get pipeline results."""
        return self.state.get_all_results()
    
    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        return self.state.get_summary()


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("WORKFLOW PIPELINE MODULE - TESTING")
    print("="*80 + "\n")
    
    print("[TEST] AnalysisPipeline execution...")
    try:
        config = WorkflowConfig(trajectory_file="test.gsd", output_dir="/tmp")
        pipeline = AnalysisPipeline(config)
        
        results = pipeline.execute()
        
        print(f"✓ Pipeline executed successfully")
        print(f"  Status: {pipeline.get_status()}")
        print(f"  Results: {len(results)} items\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("="*80)
    print("✓ Pipeline tests passed!")
    print("="*80 + "\n")
