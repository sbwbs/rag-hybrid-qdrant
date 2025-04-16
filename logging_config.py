import logging
import logging.handlers
import os
from datetime import datetime

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Base logging configuration
def setup_logger(name: str, log_file: str, level=logging.DEBUG):
    """Setup a logger with file and console handlers"""
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Create file handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Only show INFO and above in console
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Setup loggers for each component
def setup_all_loggers():
    """Setup loggers for all components"""
    
    # Get current timestamp for log filenames
    timestamp = datetime.now().strftime('%Y%m%d')
    
    # Setup loggers
    app_logger = setup_logger(
        'app',
        f'logs/app_{timestamp}.log'
    )
    
    search_engine_logger = setup_logger(
        'search_engine',
        f'logs/search_engine_{timestamp}.log'
    )
    
    document_processor_logger = setup_logger(
        'document_processor',
        f'logs/document_processor_{timestamp}.log'
    )
    
    return {
        'app': app_logger,
        'search_engine': search_engine_logger,
        'document_processor': document_processor_logger
    } 