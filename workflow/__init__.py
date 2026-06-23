"""
High-level workflow stages for the modular contact-analysis project.

Overview
--------
This module serves as the public API for the workflow package. It exposes
high-level orchestration functions and result types for the two main stages
of the contact-analysis workflow:

    1. Shape Preparation Stage
       Loads and categorizes the reference particle geometry
    2. Contact Analysis Stage
       Processes trajectory frames and computes contact metrics

Module Organization
-------------------
The workflow package is organized into logical stages:

    shape_stage.py
        Prepares the reference particle shape geometry.
        Contains: prepare_shape_geometry()
    
    contact_stage.py
        Runs the main contact-analysis workflow on trajectory data.
        Contains: run_contact_analysis() and related utilities

Design Philosophy
-----------------
Each stage is designed to be modular and testable independently:

    - Shape Stage: Pure geometry processing; no trajectory dependencies
    - Contact Stage: Uses prepared geometry to analyze frames

Public API
----------
This module exports:

    Shape Preparation API:
        ShapePreparationResult
            Named tuple containing prepared geometry data
        prepare_shape_geometry()
            Main entry point for shape preparation

    Contact Analysis API:
        ContactAnalysisRunResult
            Named tuple containing analysis results
        FrameNeighborContext
            Named tuple for per-frame neighbor information
        ParticleContactResult
            Named tuple for per-particle contact data
        iter_neighbor_frames()
            Generator function for distributed frame processing
        analyze_particle_contacts()
            Analyzes contacts for a single particle
        run_contact_analysis()
            Main entry point for the complete workflow

Typical Usage Pattern
---------------------
The workflow is typically invoked as follows:

    from workflow import (
        prepare_shape_geometry,
        run_contact_analysis,
    )

    # Stage 1: Prepare geometry (done once)
    shape_geometry = prepare_shape_geometry(
        shape_file="shape.json",
        num_edges=12,
        num_faces=6,
        verbose=True,
    )

    # Stage 2: Run contact analysis (main computation)
    result = run_contact_analysis(
        params=config,
        shape_geometry=shape_geometry,
        comm=mpi_comm,
        rank=mpi_rank,
        size=mpi_size,
        verbose=True,
        write_outputs=True,
        save_plots=True,
    )

    # Access results
    print(f"Frames processed: {result.frames_processed}")
    print(f"Contact records: {len(result.pair_records)}")

Dependencies
------------
The workflow module depends on several other project modules:

    - config : Parameter loading and validation
    - dataio : Input/output helpers for JSON and trajectories
    - geometry : Low-level geometric computations
    - parallel : MPI utilities and rank-aware printing

These dependencies are imported internally by the workflow stages.
"""

from .shape_stage import (
    ShapePreparationResult,
    prepare_shape_geometry,
)


__all__ = [
    "ShapePreparationResult",
    "prepare_shape_geometry",
]

from .contact_stage import (
    ContactAnalysisRunResult,
    FrameNeighborContext,
    ParticleContactResult,
    analyze_particle_contacts,
    iter_neighbor_frames,
    run_contact_analysis,
)

__all__ += [
    "ContactAnalysisRunResult",
    "FrameNeighborContext",
    "ParticleContactResult",
    "iter_neighbor_frames",
    "analyze_particle_contacts",
    "run_contact_analysis",
]
