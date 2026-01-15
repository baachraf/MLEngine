"""
Visualization Style Management
==============================

Functions for managing plot aesthetics, themes, and configurations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional

# Default configuration
DEFAULT_CONFIG = {
    'figure.figsize': (10, 6),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],
}

def set_style(style: str = 'whitegrid', context: str = 'notebook', 
              palette: str = 'deep', font_scale: float = 1.0, 
              rc: Optional[Dict[str, Any]] = None):
    """
    Set the global plotting style and aesthetics.
    
    Parameters
    ----------
    style : str, default='whitegrid'
        Seaborn style ('darkgrid', 'whitegrid', 'dark', 'white', 'ticks').
    context : str, default='notebook'
        Seaborn context ('paper', 'notebook', 'talk', 'poster').
    palette : str, default='deep'
        Color palette name.
    font_scale : float, default=1.0
        Scaling factor for font sizes.
    rc : dict, optional
        Dictionary of rc parameters to override defaults.
    """
    # Apply seaborn style and context
    sns.set_theme(style=style, context=context, font_scale=font_scale, rc=rc)
    
    # Set color palette
    sns.set_palette(palette)
    
    # Update matplotlib rcParams with defaults if not already set by seaborn
    for key, value in DEFAULT_CONFIG.items():
        if key not in plt.rcParams:
            plt.rcParams[key] = value

def get_current_style() -> Dict[str, Any]:
    """
    Get the current matplotlib rcParams.
    
    Returns
    -------
    dict
        Current configuration dictionary.
    """
    return dict(plt.rcParams)

def reset_style():
    """Reset matplotlib parameters to defaults."""
    plt.rcdefaults()
