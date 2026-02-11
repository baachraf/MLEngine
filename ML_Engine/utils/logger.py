"""
Logger Configuration
====================

Configures a central logger for the ML_Engine library.
"""

import logging
import sys

def get_logger(name: str = "ML_Engine", level: int = logging.INFO) -> logging.Logger:
    """
    Get a configured logger instance.

    Parameters
    ----------
    name : str, default="ML_Engine"
        The name of the logger.
    level : int, default=logging.INFO
        The logging level (e.g., logging.INFO, logging.DEBUG).

    Returns
    -------
    logging.Logger
        A configured logger instance.
    """
    # Create a logger
    logger = logging.getLogger(name)
    
    # Avoid adding duplicate handlers if the logger is already configured
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(level)

    # Create a handler to write to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    return logger

# Example of how to use it in other modules:
# from .logger import get_logger
# logger = get_logger(__name__)
# logger.info("This is an info message.")
# logger.warning("This is a warning.")
