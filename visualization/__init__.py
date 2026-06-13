# visualization/__init__.py
# ==============================================================================
# Module: visualization
# Purpose: Plotting and output domain - RDF, histograms, styling, interactive
#
# Exports:
#   - RDFPlotter, ContactMapPlotter (plots)
#   - HistogramPlotter, MultiHistogramPlotter (histograms)
#   - ColorPalette, PlotStyle, AxisLabeler (styling)
#   - InteractivePlotter, PlotInteractor (interactive)
#
# Author: Contact Analysis Team
# ==============================================================================

from .plots import RDFPlotter, ContactMapPlotter
from .histograms import HistogramPlotter, MultiHistogramPlotter
from .styling import ColorPalette, PlotStyle, AxisLabeler
from .interactive import InteractivePlotter, PlotInteractor

__all__ = [
    # Plots
    'RDFPlotter',
    'ContactMapPlotter',
    
    # Histograms
    'HistogramPlotter',
    'MultiHistogramPlotter',
    
    # Styling
    'ColorPalette',
    'PlotStyle',
    'AxisLabeler',
    
    # Interactive
    'InteractivePlotter',
    'PlotInteractor',
]

__version__ = '1.0.0'
__author__ = 'Contact Analysis Team'
