"""
Visualization Sub-package
=========================

A comprehensive toolkit for creating data science and model evaluation plots.

This package provides modules for different plot types:
- distribution: For visualizing single variables.
- relationship: For visualizing relationships between two or more variables.
- categorical: For visualizing categorical data.
- matrix: For matrix-style plots like heatmaps.
- evaluation: For model performance plots.
- style: For managing plot aesthetics.
"""

from . import style
from . import distribution
from . import relationship
from . import categorical
from . import matrix
from . import evaluation
