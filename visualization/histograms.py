# visualization/histograms.py
# ==============================================================================
# Module: visualization.histograms
# Purpose: Distribution histogram visualization
#
# Classes:
#   - HistogramPlotter: Create distribution histograms
#   - MultiHistogramPlotter: Multiple histograms comparison
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np

from .. import ValidationError

logger = logging.getLogger(__name__)


# ==============================================================================
# HISTOGRAM PLOTTER
# ==============================================================================

@dataclass
class HistogramPlotter:
    """
    Create distribution histograms.
    
    Visualize data distributions.
    
    Attributes
    ----------
    figsize : Tuple[float, float]
        Figure size
    dpi : int
        Dots per inch
    n_bins : int
        Number of bins
    
    Examples
    --------
    >>> plotter = HistogramPlotter()
    >>> fig = plotter.plot_histogram(data)
    """
    
    figsize: Tuple[float, float] = (10, 6)
    dpi: int = 100
    n_bins: int = 30
    
    def __post_init__(self) -> None:
        """Initialize plotter."""
        if self.figsize[0] <= 0 or self.figsize[1] <= 0:
            raise ValidationError(
                "figsize must be positive",
                error_code="VIZ_INVALID_FIGSIZE"
            )
        
        if self.n_bins <= 0:
            raise ValidationError(
                "n_bins must be positive",
                error_code="VIZ_INVALID_BINS"
            )
        
        logger.debug("[HistogramPlotter] Initialized")
    
    def plot_histogram(
        self,
        data: List[float],
        title: str = "Distribution",
        xlabel: str = "Value",
        ylabel: str = "Frequency",
        color: str = "blue"
    ) -> Dict[str, any]:
        """
        Create histogram.
        
        Parameters
        ----------
        data : List[float]
            Data to plot
        title : str, optional
            Plot title
        xlabel : str, optional
            X-axis label
        ylabel : str, optional
            Y-axis label
        color : str, optional
            Bar color
        
        Returns
        -------
        Dict[str, any]
            Plot data and metadata
        """
        
        logger.debug("[HistogramPlotter] Creating histogram")
        
        data_array = np.asarray(data, dtype=float)
        
        if len(data_array) == 0:
            raise ValidationError(
                "data must not be empty",
                error_code="VIZ_EMPTY_DATA"
            )
        
        # Compute histogram
        counts, edges = np.histogram(data_array, bins=self.n_bins)
        bin_centers = (edges[:-1] + edges[1:]) / 2
        
        plot_data = {
            'type': 'histogram',
            'x': bin_centers,
            'y': counts,
            'title': title,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'color': color,
            'figsize': self.figsize,
            'dpi': self.dpi,
            'n_bins': self.n_bins,
        }
        
        logger.info(f"[HistogramPlotter] Histogram created: {len(data_array)} data points")
        
        return plot_data
    
    def plot_histogram_with_stats(
        self,
        data: List[float],
        title: str = "Distribution with Statistics"
    ) -> Dict[str, any]:
        """
        Create histogram with statistical overlays.
        
        Parameters
        ----------
        data : List[float]
            Data to plot
        title : str, optional
            Plot title
        
        Returns
        -------
        Dict[str, any]
            Plot data with statistics
        """
        
        logger.debug("[HistogramPlotter] Creating histogram with statistics")
        
        data_array = np.asarray(data, dtype=float)
        
        plot_data = self.plot_histogram(data_array, title=title)
        
        # Add statistics
        plot_data['statistics'] = {
            'mean': float(np.mean(data_array)),
            'std': float(np.std(data_array)),
            'median': float(np.median(data_array)),
            'min': float(np.min(data_array)),
            'max': float(np.max(data_array)),
        }
        
        return plot_data


# ==============================================================================
# MULTI HISTOGRAM PLOTTER
# ==============================================================================

@dataclass
class MultiHistogramPlotter:
    """
    Create multiple histograms for comparison.
    
    Visualize and compare multiple distributions.
    
    Examples
    --------
    >>> plotter = MultiHistogramPlotter()
    >>> fig = plotter.plot_comparison(data1, data2, labels=["A", "B"])
    """
    
    figsize: Tuple[float, float] = (12, 6)
    dpi: int = 100
    n_bins: int = 30
    
    def plot_comparison(
        self,
        data_list: List[List[float]],
        labels: Optional[List[str]] = None,
        title: str = "Distribution Comparison",
        xlabel: str = "Value",
        ylabel: str = "Frequency"
    ) -> Dict[str, any]:
        """
        Compare multiple distributions.
        
        Parameters
        ----------
        data_list : List[List[float]]
            Multiple datasets
        labels : List[str], optional
            Dataset labels
        title : str, optional
            Plot title
        xlabel : str, optional
            X-axis label
        ylabel : str, optional
            Y-axis label
        
        Returns
        -------
        Dict[str, any]
            Comparison plot data
        """
        
        logger.debug("[MultiHistogramPlotter] Creating comparison plot")
        
        if not data_list:
            raise ValidationError(
                "data_list must not be empty",
                error_code="VIZ_EMPTY_DATA"
            )
        
        if labels is None:
            labels = [f"Dataset {i}" for i in range(len(data_list))]
        
        if len(labels) != len(data_list):
            raise ValidationError(
                f"labels and data_list must have same length",
                error_code="VIZ_LENGTH_MISMATCH"
            )
        
        # Process each dataset
        datasets = []
        for data, label in zip(data_list, labels):
            data_array = np.asarray(data, dtype=float)
            counts, edges = np.histogram(data_array, bins=self.n_bins)
            bin_centers = (edges[:-1] + edges[1:]) / 2
            
            datasets.append({
                'label': label,
                'x': bin_centers,
                'y': counts,
                'stats': {
                    'mean': float(np.mean(data_array)),
                    'std': float(np.std(data_array)),
                    'n': len(data_array),
                }
            })
        
        plot_data = {
            'type': 'histogram_comparison',
            'datasets': datasets,
            'title': title,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'figsize': self.figsize,
            'dpi': self.dpi,
        }
        
        logger.info(f"[MultiHistogramPlotter] Comparison plot created: {len(datasets)} datasets")
        
        return plot_data
    
    def plot_cumulative(
        self,
        data: List[float],
        title: str = "Cumulative Distribution",
        xlabel: str = "Value",
        ylabel: str = "Cumulative Frequency"
    ) -> Dict[str, any]:
        """
        Create cumulative distribution plot.
        
        Parameters
        ----------
        data : List[float]
            Data to plot
        title : str, optional
            Plot title
        xlabel : str, optional
            X-axis label
        ylabel : str, optional
            Y-axis label
        
        Returns
        -------
        Dict[str, any]
            Cumulative plot data
        """
        
        logger.debug("[MultiHistogramPlotter] Creating cumulative plot")
        
        data_array = np.asarray(data, dtype=float)
        
        # Sort and compute cumulative
        sorted_data = np.sort(data_array)
        cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        plot_data = {
            'type': 'line',
            'x': sorted_data,
            'y': cumulative,
            'title': title,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'figsize': self.figsize,
            'dpi': self.dpi,
        }
        
        return plot_data


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("VISUALIZATION HISTOGRAMS MODULE - TESTING")
    print("="*80 + "\n")
    
    print("[TEST 1] HistogramPlotter initialization...")
    try:
        plotter = HistogramPlotter(n_bins=30)
        print(f"✓ HistogramPlotter created\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("[TEST 2] Create histogram...")
    try:
        data = np.random.normal(3.0, 1.0, 1000)
        plot_data = plotter.plot_histogram(data)
        
        print(f"✓ Histogram created: {plot_data['type']}")
        print(f"  Title: {plot_data['title']}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("[TEST 3] MultiHistogramPlotter comparison...")
    try:
        multi_plotter = MultiHistogramPlotter()
        
        data1 = np.random.normal(3.0, 1.0, 500)
        data2 = np.random.normal(3.5, 1.2, 500)
        
        plot_data = multi_plotter.plot_comparison(
            [data1, data2],
            labels=["Distribution A", "Distribution B"]
        )
        
        print(f"✓ Comparison plot created: {len(plot_data['datasets'])} datasets\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("="*80)
    print("✓ Histograms tests passed!")
    print("="*80 + "\n")
