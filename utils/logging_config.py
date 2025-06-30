import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO", 
                 format_string: Optional[str] = None,
                 include_timestamp: bool = True) -> logging.Logger:
    """
    Setup logging configuration for the application
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
        include_timestamp: Whether to include timestamp in log messages
        
    Returns:
        Configured logger instance
    """
    if format_string is None:
        if include_timestamp:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        else:
            format_string = '%(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger('deepgrape')
    return logger


def get_logger(name: str = 'deepgrape') -> logging.Logger:
    """Get a logger instance with the specified name"""
    return logging.getLogger(name)
