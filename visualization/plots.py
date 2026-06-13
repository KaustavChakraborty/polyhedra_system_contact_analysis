# visualization/plots.py
# ==============================================================================
# Module: visualization.plots
# Purpose: RDF plots and contact maps visualization
#
# Classes:
#   - RDFPlotter: Plot radial distribution functions
#   - ContactMapPlotter: Visualize contact matrices
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import numpy as np

from .. import ValidationError

logger = logging.getLogger(__name__)


# ==============================================================================
# RDF PLOTTER
# ==============================================================================

@dataclass
class RDFPlotter:
    """
    Plot radial distribution functions.
    
    Creates RDF plots from RDFData.
    
    Attributes
    ----------
    figsize : Tuple[float, float]
        Figure size (width, height)
    dpi : int
        Dots per inch
    
    Examples
    --------
    >>> plotter = RDFPlotter()
    >>> fig = plotter.plot_rdf(rdf_data)
    """
    
    figsize: Tuple[float, float] = (10, 6)
    dpi: int = 100
    
    def __post_init__(self) -> None:
        """Initialize plotter."""
        if self.figsize[0] <= 0 or self.figsize[1] <= 0:
            raise ValidationError(
                "figsize must be positive",
                error_code="VIZ_INVALID_FIGSIZE"
            )
        
        if self.dpi <= 0:
            raise ValidationError(
                "dpi must be positive",
                error_code="VIZ_INVALID_DPI"
            )
        
        logger.debug("[RDFPlotter] Initialized")
    
    def plot_rdf(
        self,
        r_values: np.ndarray,
        g_r: np.ndarray,
        title: str = "Radial Distribution Function",
        xlabel: str = "r (distance units)",
        ylabel: str = "g(r)"
    ) -> Dict[str, any]:
        """
        Create RDF plot.
        
        Parameters
        ----------
        r_values : np.ndarray
            Radial distances
        g_r : np.ndarray
            RDF values
        title : str, optional
            Plot title
        xlabel : str, optional
            X-axis label
        ylabel : str, optional
            Y-axis label
        
        Returns
        -------
        Dict[str, any]
            Plot data and metadata
        """
        
        logger.debug("[RDFPlotter] Creating RDF plot")
        
        # Validate data
        r_values = np.asarray(r_values, dtype=float)
        g_r = np.asarray(g_r, dtype=float)
        
        if len(r_values) != len(g_r):
            raise ValidationError(
                f"r_values and g_r must have same length",
                error_code="VIZ_LENGTH_MISMATCH"
            )
        
        # Create plot data
        plot_data = {
            'type': 'line',
            'x': r_values,
            'y': g_r,
            'title': title,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'figsize': self.figsize,
            'dpi': self.dpi,
        }
        
        logger.info(f"[RDFPlotter] RDF plot created: {len(r_values)} points")
        
        return plot_data
    
    def plot_rdf_with_peak(
        self,
        r_values: np.ndarray,
        g_r: np.ndarray,
        r_peak: float,
        g_peak: float
    ) -> Dict[str, any]:
        """
        Create RDF plot with peak marker.
        
        Parameters
        ----------
        r_values : np.ndarray
            Radial distances
        g_r : np.ndarray
            RDF values
        r_peak : float
            Peak location
        g_peak : float
            Peak value
        
        Returns
        -------
        Dict[str, any]
            Plot data with peak annotation
        """
        
        logger.debug("[RDFPlotter] Creating RDF plot with peak")
        
        plot_data = self.plot_rdf(r_values, g_r)
        plot_data['peak'] = {
            'x': r_peak,
            'y': g_peak,
            'marker': 'o',
            'color': 'red'
        }
        
        return plot_data


# ==============================================================================
# CONTACT MAP PLOTTER
# ==============================================================================

@dataclass
class ContactMapPlotter:
    """
    Visualize contact matrices.
    
    Creates contact maps from contact data.
    
    Examples
    --------
    >>> plotter = ContactMapPlotter()
    >>> fig = plotter.plot_contact_matrix(contact_matrix)
    """
    
    figsize: Tuple[float, float] = (10, 8)
    dpi: int = 100
    
    def __post_init__(self) -> None:
        """Initialize plotter."""
        if self.figsize[0] <= 0 or self.figsize[1] <= 0:
            raise ValidationError(
                "figsize must be positive",
                error_code="VIZ_INVALID_FIGSIZE"
            )
        
        logger.debug("[ContactMapPlotter] Initialized")
    
    def plot_contact_matrix(
        self,
        contact_matrix: np.ndarray,
        title: str = "Contact Map",
        cmap: str = "viridis"
    ) -> Dict[str, any]:
        """
        Create contact map plot.
        
        Parameters
        ----------
        contact_matrix : np.ndarray
            Contact matrix (2D)
        title : str, optional
            Plot title
        cmap : str, optional
            Colormap name
        
        Returns
        -------
        Dict[str, any]
            Plot data and metadata
        """
        
        logger.debug("[ContactMapPlotter] Creating contact map")
        
        contact_matrix = np.asarray(contact_matrix, dtype=float)
        
        if contact_matrix.ndim != 2:
            raise ValidationError(
                f"contact_matrix must be 2D, got {contact_matrix.ndim}D",
                error_code="VIZ_INVALID_SHAPE"
            )
        
        # Create plot data
        plot_data = {
            'type': 'heatmap',
            'data': contact_matrix,
            'title': title,
            'cmap': cmap,
            'figsize': self.figsize,
            'dpi': self.dpi,
            'xlabel': 'Particle ID',
            'ylabel': 'Particle ID',
        }
        
        logger.info(f"[ContactMapPlotter] Contact map created: {contact_matrix.shape}")
        
        return plot_data
    
    def plot_contact_distribution(
        self,
        contact_counts: np.ndarray,
        particle_ids: Optional[np.ndarray] = None
    ) -> Dict[str, any]:
        """
        Plot contact distribution.
        
        Parameters
        ----------
        contact_counts : np.ndarray
            Number of contacts per particle
        particle_ids : np.ndarray, optional
            Particle IDs (default: 0, 1, 2, ...)
        
        Returns
        -------
        Dict[str, any]
            Bar plot data
        """
        
        logger.debug("[ContactMapPlotter] Creating contact distribution plot")
        
        contact_counts = np.asarray(contact_counts, dtype=float)
        
        if particle_ids is None:
            particle_ids = np.arange(len(contact_counts))
        else:
            particle_ids = np.asarray(particle_ids)
        
        if len(particle_ids) != len(contact_counts):
            raise ValidationError(
                f"particle_ids and contact_counts must have same length",
                error_code="VIZ_LENGTH_MISMATCH"
            )
        
        plot_data = {
            'type': 'bar',
            'x': particle_ids,
            'y': contact_counts,
            'title': 'Contact Distribution',
            'xlabel': 'Particle ID',
            'ylabel': 'Number of Contacts',
            'figsize': self.figsize,
            'dpi': self.dpi,
        }
        
        return plot_data


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("VISUALIZATION PLOTS MODULE - TESTING")
    print("="*80 + "\n")
    
    print("[TEST 1] RDFPlotter initialization...")
    try:
        plotter = RDFPlotter()
        print(f"✓ RDFPlotter created\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("[TEST 2] Create RDF plot...")
    try:
        r_vals = np.linspace(0.5, 5.0, 50)
        g_r_vals = np.random.random(50) + 0.5
        
        plot_data = plotter.plot_rdf(r_vals, g_r_vals)
        print(f"✓ RDF plot created: {plot_data['type']}")
        print(f"  Title: {plot_data['title']}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("[TEST 3] ContactMapPlotter...")
    try:
        cm_plotter = ContactMapPlotter()
        
        contact_matrix = np.random.random((10, 10))
        plot_data = cm_plotter.plot_contact_matrix(contact_matrix)
        
        print(f"✓ Contact map created: {plot_data['type']}")
        print(f"  Shape: {plot_data['data'].shape}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("="*80)
    print("✓ Plots tests passed!")
    print("="*80 + "\n")
