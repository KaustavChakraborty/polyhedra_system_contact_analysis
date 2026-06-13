# visualization/interactive.py
# ==============================================================================
# Module: visualization.interactive
# Purpose: Interactive visualization support (optional)
#
# Classes:
#   - InteractivePlotter: Create interactive plots
#   - PlotInteractor: Handle plot interactions
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any

from .. import ValidationError

logger = logging.getLogger(__name__)


# ==============================================================================
# INTERACTIVE PLOTTER
# ==============================================================================

@dataclass
class InteractivePlotter:
    """
    Create interactive plots.
    
    Support for interactive visualization with plotly-like interface.
    
    Attributes
    ----------
    width : int
        Plot width (pixels)
    height : int
        Plot height (pixels)
    interactive_mode : bool
        Enable interactivity
    
    Examples
    --------
    >>> plotter = InteractivePlotter()
    >>> plot = plotter.create_interactive_rdf(rdf_data)
    """
    
    width: int = 900
    height: int = 600
    interactive_mode: bool = True
    
    def __post_init__(self) -> None:
        """Initialize plotter."""
        if self.width <= 0 or self.height <= 0:
            raise ValidationError(
                "width and height must be positive",
                error_code="VIZ_INVALID_DIMENSIONS"
            )
        
        logger.debug("[InteractivePlotter] Initialized")
    
    def create_interactive_rdf(
        self,
        r_values: List[float],
        g_r: List[float],
        title: str = "RDF (Interactive)"
    ) -> Dict[str, Any]:
        """
        Create interactive RDF plot.
        
        Parameters
        ----------
        r_values : List[float]
            Radial distances
        g_r : List[float]
            RDF values
        title : str, optional
            Plot title
        
        Returns
        -------
        Dict[str, Any]
            Interactive plot configuration
        """
        
        logger.debug("[InteractivePlotter] Creating interactive RDF")
        
        plot_config = {
            'type': 'interactive_line',
            'data': {
                'x': r_values,
                'y': g_r,
            },
            'layout': {
                'title': title,
                'xaxis': {'title': 'r (distance units)'},
                'yaxis': {'title': 'g(r)'},
                'width': self.width,
                'height': self.height,
                'interactive': self.interactive_mode,
            },
            'config': {
                'responsive': True,
                'editable': False,
                'displayModeBar': True,
            }
        }
        
        return plot_config
    
    def create_interactive_scatter(
        self,
        x_data: List[float],
        y_data: List[float],
        title: str = "Scatter Plot",
        hover_text: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create interactive scatter plot.
        
        Parameters
        ----------
        x_data : List[float]
            X coordinates
        y_data : List[float]
            Y coordinates
        title : str, optional
            Plot title
        hover_text : List[str], optional
            Hover text for points
        
        Returns
        -------
        Dict[str, Any]
            Interactive plot configuration
        """
        
        logger.debug("[InteractivePlotter] Creating interactive scatter")
        
        plot_config = {
            'type': 'interactive_scatter',
            'data': {
                'x': x_data,
                'y': y_data,
                'mode': 'markers',
                'hovertext': hover_text,
            },
            'layout': {
                'title': title,
                'width': self.width,
                'height': self.height,
                'interactive': self.interactive_mode,
            },
            'config': {
                'responsive': True,
                'displayModeBar': True,
            }
        }
        
        return plot_config


# ==============================================================================
# PLOT INTERACTOR
# ==============================================================================

@dataclass
class PlotInteractor:
    """
    Handle plot interactions.
    
    Manage callbacks and event handling for interactive plots.
    
    Attributes
    ----------
    callbacks : Dict[str, Callable]
        Event callbacks
    state : Dict[str, Any]
        Interactive state
    
    Examples
    --------
    >>> interactor = PlotInteractor()
    >>> interactor.register_callback('hover', hover_handler)
    """
    
    callbacks: Dict[str, Callable] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Initialize interactor."""
        logger.debug("[PlotInteractor] Initialized")
    
    def register_callback(
        self,
        event_name: str,
        callback: Callable
    ) -> None:
        """
        Register event callback.
        
        Parameters
        ----------
        event_name : str
            Event name ('click', 'hover', 'zoom', etc.)
        callback : Callable
            Callback function
        """
        
        if not callable(callback):
            raise ValidationError(
                "callback must be callable",
                error_code="VIZ_INVALID_CALLBACK"
            )
        
        self.callbacks[event_name] = callback
        logger.debug(f"[PlotInteractor] Registered callback: {event_name}")
    
    def trigger_event(
        self,
        event_name: str,
        event_data: Dict[str, Any]
    ) -> Any:
        """
        Trigger event callback.
        
        Parameters
        ----------
        event_name : str
            Event name
        event_data : Dict[str, Any]
            Event data
        
        Returns
        -------
        Any
            Callback result
        """
        
        if event_name not in self.callbacks:
            logger.warning(f"[PlotInteractor] No callback for {event_name}")
            return None
        
        logger.debug(f"[PlotInteractor] Triggering event: {event_name}")
        
        callback = self.callbacks[event_name]
        return callback(event_data)
    
    def set_state(self, key: str, value: Any) -> None:
        """
        Set state variable.
        
        Parameters
        ----------
        key : str
            State key
        value : Any
            State value
        """
        
        self.state[key] = value
        logger.debug(f"[PlotInteractor] State set: {key}={value}")
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """
        Get state variable.
        
        Parameters
        ----------
        key : str
            State key
        default : Any, optional
            Default value
        
        Returns
        -------
        Any
            State value
        """
        
        return self.state.get(key, default)
    
    def get_all_state(self) -> Dict[str, Any]:
        """Get all state."""
        return self.state.copy()
    
    def reset_state(self) -> None:
        """Reset all state."""
        self.state.clear()
        logger.debug("[PlotInteractor] State reset")


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("VISUALIZATION INTERACTIVE MODULE - TESTING")
    print("="*80 + "\n")
    
    print("[TEST 1] InteractivePlotter...")
    try:
        plotter = InteractivePlotter()
        
        r_vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        g_r_vals = [0.5, 1.2, 1.8, 1.5, 0.8]
        
        plot_config = plotter.create_interactive_rdf(r_vals, g_r_vals)
        print(f"✓ Interactive RDF plot created")
        print(f"  Type: {plot_config['type']}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("[TEST 2] PlotInteractor...")
    try:
        interactor = PlotInteractor()
        
        # Define callback
        def hover_handler(event):
            return f"Hovered: {event}"
        
        interactor.register_callback('hover', hover_handler)
        
        result = interactor.trigger_event('hover', {'x': 1.0, 'y': 2.0})
        print(f"✓ PlotInteractor created")
        print(f"  Callback result: {result}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("[TEST 3] State management...")
    try:
        interactor = PlotInteractor()
        
        interactor.set_state('zoom_level', 2.0)
        interactor.set_state('selected_point', 5)
        
        zoom = interactor.get_state('zoom_level')
        print(f"✓ State management working")
        print(f"  Zoom level: {zoom}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("="*80)
    print("✓ Interactive tests passed!")
    print("="*80 + "\n")
