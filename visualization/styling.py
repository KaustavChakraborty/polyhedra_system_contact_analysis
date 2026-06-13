# visualization/styling.py
# ==============================================================================
# Module: visualization.styling
# Purpose: Plot themes, colors, and styling
#
# Classes:
#   - ColorPalette: Color scheme definitions
#   - PlotStyle: Plot styling templates
#   - AxisLabeler: Label formatting
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

from .. import ValidationError

logger = logging.getLogger(__name__)


# ==============================================================================
# COLOR PALETTE
# ==============================================================================

@dataclass
class ColorPalette:
    """
    Color scheme definitions.
    
    Manages consistent colors across plots.
    
    Attributes
    ----------
    primary : str
        Primary color (hex)
    secondary : str
        Secondary color (hex)
    accent : str
        Accent color (hex)
    colors : Dict[str, str]
        Named colors
    
    Examples
    --------
    >>> palette = ColorPalette()
    >>> color = palette.get_color("rdf")
    """
    
    primary: str = "#1f77b4"
    secondary: str = "#ff7f0e"
    accent: str = "#2ca02c"
    colors: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Initialize palette."""
        if not self.colors:
            self.colors = {
                'rdf': "#1f77b4",
                'histogram': "#ff7f0e",
                'contact': "#2ca02c",
                'statistics': "#d62728",
                'background': "#ffffff",
                'grid': "#e0e0e0",
                'text': "#333333",
            }
        
        logger.debug("[ColorPalette] Initialized")
    
    def get_color(self, name: str) -> str:
        """
        Get color by name.
        
        Parameters
        ----------
        name : str
            Color name
        
        Returns
        -------
        str
            Color hex value
        """
        
        return self.colors.get(name, self.primary)
    
    def set_color(self, name: str, hex_color: str) -> None:
        """
        Set color.
        
        Parameters
        ----------
        name : str
            Color name
        hex_color : str
            Hex color value
        """
        
        if not hex_color.startswith('#') or len(hex_color) != 7:
            raise ValidationError(
                f"Invalid hex color: {hex_color}",
                error_code="VIZ_INVALID_COLOR"
            )
        
        self.colors[name] = hex_color
        logger.debug(f"[ColorPalette] Set {name} to {hex_color}")
    
    def get_palette(self) -> Dict[str, str]:
        """Get all colors."""
        return self.colors.copy()


# ==============================================================================
# PLOT STYLE
# ==============================================================================

@dataclass
class PlotStyle:
    """
    Plot styling template.
    
    Defines consistent styling for plots.
    
    Attributes
    ----------
    name : str
        Style name
    font_family : str
        Font family
    font_size : int
        Base font size
    linewidth : float
        Line width
    markersize : float
        Marker size
    palette : ColorPalette
        Color palette
    
    Examples
    --------
    >>> style = PlotStyle()
    >>> config = style.get_config()
    """
    
    name: str = "default"
    font_family: str = "sans-serif"
    font_size: int = 12
    linewidth: float = 2.0
    markersize: float = 8.0
    palette: ColorPalette = None
    
    def __post_init__(self) -> None:
        """Initialize style."""
        if self.font_size <= 0:
            raise ValidationError(
                "font_size must be positive",
                error_code="VIZ_INVALID_FONT_SIZE"
            )
        
        if self.linewidth <= 0:
            raise ValidationError(
                "linewidth must be positive",
                error_code="VIZ_INVALID_LINEWIDTH"
            )
        
        if self.palette is None:
            self.palette = ColorPalette()
        
        logger.debug(f"[PlotStyle] Initialized: {self.name}")
    
    def get_config(self) -> Dict[str, any]:
        """
        Get style configuration.
        
        Returns
        -------
        Dict[str, any]
            Style configuration
        """
        
        return {
            'name': self.name,
            'font_family': self.font_family,
            'font_size': self.font_size,
            'linewidth': self.linewidth,
            'markersize': self.markersize,
            'colors': self.palette.get_palette(),
        }
    
    @staticmethod
    def get_preset(preset_name: str) -> PlotStyle:
        """
        Get preset style.
        
        Parameters
        ----------
        preset_name : str
            Style name: 'default', 'seaborn', 'minimal'
        
        Returns
        -------
        PlotStyle
            Configured style
        """
        
        logger.debug(f"[PlotStyle] Loading preset: {preset_name}")
        
        presets = {
            'default': PlotStyle(name='default'),
            'seaborn': PlotStyle(
                name='seaborn',
                font_family='sans-serif',
                font_size=11,
                linewidth=1.5,
            ),
            'minimal': PlotStyle(
                name='minimal',
                font_family='serif',
                font_size=10,
                linewidth=1.0,
            ),
        }
        
        if preset_name not in presets:
            logger.warning(f"Unknown preset: {preset_name}, using default")
            return presets['default']
        
        return presets[preset_name]


# ==============================================================================
# AXIS LABELER
# ==============================================================================

@dataclass
class AxisLabeler:
    """
    Format axis labels.
    
    Provides consistent label formatting.
    
    Examples
    --------
    >>> labeler = AxisLabeler()
    >>> label = labeler.format_label("distance", unit="nm")
    """
    
    def format_label(
        self,
        quantity: str,
        unit: Optional[str] = None,
        latex: bool = False
    ) -> str:
        """
        Format axis label.
        
        Parameters
        ----------
        quantity : str
            Quantity name
        unit : str, optional
            Unit name
        latex : bool, optional
            Use LaTeX formatting
        
        Returns
        -------
        str
            Formatted label
        """
        
        if latex:
            label = f"${quantity}$"
        else:
            label = quantity
        
        if unit:
            if latex:
                label += f" ({unit})"
            else:
                label += f" ({unit})"
        
        return label
    
    def format_title(
        self,
        title: str,
        subtitle: Optional[str] = None
    ) -> str:
        """
        Format plot title.
        
        Parameters
        ----------
        title : str
            Main title
        subtitle : str, optional
            Subtitle
        
        Returns
        -------
        str
            Formatted title
        """
        
        if subtitle:
            return f"{title}\n{subtitle}"
        
        return title
    
    def format_legend_label(
        self,
        name: str,
        count: Optional[int] = None
    ) -> str:
        """
        Format legend label.
        
        Parameters
        ----------
        name : str
            Item name
        count : int, optional
            Item count
        
        Returns
        -------
        str
            Formatted label
        """
        
        if count is not None:
            return f"{name} (n={count})"
        
        return name
    
    @staticmethod
    def get_standard_labels() -> Dict[str, Dict[str, str]]:
        """
        Get standard labels for quantities.
        
        Returns
        -------
        Dict[str, Dict[str, str]]
            Standard labels by quantity
        """
        
        return {
            'distance': {
                'label': 'Distance',
                'unit': 'nm',
                'latex': 'r',
            },
            'rdf': {
                'label': 'g(r)',
                'unit': None,
                'latex': 'g(r)',
            },
            'histogram': {
                'label': 'Frequency',
                'unit': 'count',
                'latex': 'N',
            },
            'contact_area': {
                'label': 'Contact Area',
                'unit': 'nm²',
                'latex': 'A_c',
            },
        }


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("VISUALIZATION STYLING MODULE - TESTING")
    print("="*80 + "\n")
    
    print("[TEST 1] ColorPalette...")
    try:
        palette = ColorPalette()
        color = palette.get_color('rdf')
        print(f"✓ ColorPalette created")
        print(f"  RDF color: {color}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("[TEST 2] PlotStyle presets...")
    try:
        style = PlotStyle.get_preset('seaborn')
        config = style.get_config()
        print(f"✓ PlotStyle loaded: {config['name']}")
        print(f"  Font size: {config['font_size']}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("[TEST 3] AxisLabeler...")
    try:
        labeler = AxisLabeler()
        label = labeler.format_label('Distance', unit='nm')
        
        print(f"✓ AxisLabeler created")
        print(f"  Formatted label: {label}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("="*80)
    print("✓ Styling tests passed!")
    print("="*80 + "\n")
