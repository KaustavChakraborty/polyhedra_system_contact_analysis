# particles/__init__.py
# ==============================================================================
# Module: particles
# Purpose: Particle systems domain - types, loading, processing
#
# Exports:
#   - Particle, ParticleSystem (types)
#   - load_shape_from_json, create_particle, create_particles_from_arrays (loader)
#   - ParticleProcessor (processor)
#   - ParticleSystemHandler (system)
#
# Author: Contact Analysis Team
# ==============================================================================

from .types import Particle, ParticleSystem
from .loader import load_shape_from_json, create_particle, create_particles_from_arrays
from .processor import ParticleProcessor
from .system import ParticleSystemHandler

__all__ = [
    # Data types
    'Particle',
    'ParticleSystem',
    
    # Loaders and factories
    'load_shape_from_json',
    'create_particle',
    'create_particles_from_arrays',
    
    # Processors
    'ParticleProcessor',
    'ParticleSystemHandler',
]

__version__ = '1.0.0'
__author__ = 'Contact Analysis Team'
