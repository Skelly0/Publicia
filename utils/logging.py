"""
Logging configuration and utilities for Publicia
"""
import io
import sys
import time
import random
import logging
import json
from datetime import datetime

def sanitize_for_logging(text: str) -> str:
    """Remove problematic characters from the string for safe logging.
    
    This function handles Unicode characters that might cause issues with certain
    terminal encodings, especially on Windows with cp1252 encoding.
    """
    if not text:
        return ""
    
    # Replace BOM and other potentially problematic characters
    # Use the 'replace' error handler to substitute any characters that can't be encoded
    try:
        # Try to encode and decode with 'replace' error handler
        # This will replace any characters that can't be encoded in cp1252
        return text.encode('cp1252', errors='replace').decode('cp1252')
    except Exception:
        # Fallback to just removing the BOM if the above fails
        return text.replace('\ufeff', '')
    
# Custom colored formatter for logs
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for console."""
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[41m\033[37m',  # White on red background
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Get the original formatted message
        msg = super().format(record)
        # Add color based on log level if defined
        if record.levelname in self.COLORS:
            return f"{self.COLORS[record.levelname]}{msg}{self.RESET}"
        return msg

def configure_logging():
    """Set up colored logging for both file and console."""
    # Reconfigure stdout to use UTF-8 with error replacement. On Windows,
    # reassigning ``sys.stdout`` can lead to ``OSError: [Errno 22]`` when
    # flushing the stream.  ``reconfigure`` updates the existing wrapper
    # safely if available.
    try:
        enc = sys.stdout.encoding or ''
        if enc.lower() != 'utf-8':
            try:
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            except (AttributeError, ValueError):
                # Fallback for older Python versions
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        # If any unexpected issue occurs, keep stdout unchanged to avoid
        # disrupting logging completely.
        pass
        
    # Create formatters
    log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    detailed_format = '%(asctime)s - %(levelname)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s'
    file_formatter = logging.Formatter(log_format)
    detailed_formatter = logging.Formatter(detailed_format)
    console_formatter = ColoredFormatter(log_format)
    
    # Create handlers
    file_handler = logging.FileHandler('bot_detailed.log', encoding='utf-8', errors='replace')
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)

    hf_file_handler = logging.FileHandler('bot_high_fidelity.log', encoding='utf-8', errors='replace')
    hf_file_handler.setFormatter(detailed_formatter)
    hf_file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers = [file_handler, hf_file_handler, console_handler]

    return logging.getLogger(__name__)

# File to store question/response pairs as JSON Lines
QA_LOG_FILE = 'qa_log.jsonl'

from typing import Optional, Dict, Any


def log_qa_pair(
    question: str,
    response: str,
    username: str,
    channel: str = None,
    multi_turn: bool = False,
    interaction_type: str = 'message',
    context: Optional[Dict[str, Any]] = None,
    model_used: Optional[str] = None,
    temperature: Optional[float] = None,
    temperature_min: Optional[float] = None,
    temperature_base: Optional[float] = None,
    temperature_max: Optional[float] = None,
):
    """Append a question/response pair to the QA log.

    Parameters
    ----------
    question: str
        The user's question.
    response: str
        The bot's response.
    username: str
        Name of the user who asked the question.
    channel: str, optional
        Channel where the interaction happened.
    multi_turn: bool
        Whether this was part of a multi-turn conversation.
    interaction_type: str
        Type of interaction (e.g. "message" or "slash_command").
    context: dict, optional
        Additional context information about the request, such as number of
        chunks or whether images were included.
    model_used: str, optional
        The model that generated the response.
    temperature: float, optional
        The actual temperature used for the response.
    temperature_min: float, optional
        The minimum allowed temperature at the time of generation.
    temperature_base: float, optional
        The base (default) temperature.
    temperature_max: float, optional
        The maximum allowed temperature at the time of generation.
    """
    entry = {
        'timestamp': datetime.now().isoformat(),
        'username': username,
        'channel': channel,
        'interaction_type': interaction_type,
        'multi_turn': multi_turn,
        'question': sanitize_for_logging(question),
        'response': sanitize_for_logging(response)
    }
    if model_used:
        entry['model_used'] = sanitize_for_logging(model_used)
    if temperature is not None:
        entry['temperature'] = temperature
    if temperature_min is not None:
        entry['temperature_min'] = temperature_min
    if temperature_base is not None:
        entry['temperature_base'] = temperature_base
    if temperature_max is not None:
        entry['temperature_max'] = temperature_max
    if context:
        sanitized_context = {}
        for key, value in context.items():
            if isinstance(value, str):
                sanitized_context[key] = sanitize_for_logging(value)
            else:
                sanitized_context[key] = value
        entry['context'] = sanitized_context
    try:
        with open(QA_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    except Exception as log_err:
        logging.getLogger(__name__).error(f"Failed to write QA log: {log_err}")

def display_startup_banner():
    """Display super cool ASCII art banner on startup with simple search indicator."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║   ██████╗ ██╗   ██╗██████╗ ██╗     ██╗ ██████╗██╗ █████╗         ║
    ║   ██╔══██╗██║   ██║██╔══██╗██║     ██║██╔════╝██║██╔══██╗        ║
    ║   ██████╔╝██║   ██║██████╔╝██║     ██║██║     ██║███████║        ║
    ║   ██╔═══╝ ██║   ██║██╔══██╗██║     ██║██║     ██║██╔══██║        ║
    ║   ██║     ╚██████╔╝██████╔╝███████╗██║╚██████╗██║██║  ██║        ║
    ║   ╚═╝      ╚═════╝ ╚═════╝ ╚══════╝╚═╝ ╚═════╝╚═╝╚═╝  ╚═╝        ║
    ║                                                                   ║
    ║           IMPERIAL ABHUMAN MENTAT INTERFACE                       ║
    ║                                                                   ║
    ║       * Ledus Banum 77 Knowledge Repository *                     ║
    ║       * Imperial Lore Reference System *                          ║
    ║                                                                   ║
    ║       [NEURAL PATHWAY INITIALIZATION SEQUENCE]                    ║
    ║                                                                   ║
    ║       ** SIMPLE SEARCH MODE ACTIVE - ENHANCED SEARCH DISABLED **  ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """
    
    # Add color to the banner
    cyan = '\033[36m'
    reset = '\033[0m'
    print(f"{cyan}{banner}{reset}")

    # Display simulation of "neural pathway initialization"
    print(f"{cyan}[INITIATING NEURAL PATHWAYS - SIMPLE SEARCH MODE]{reset}")
    for i in range(10):
        dots = "." * random.randint(3, 10)
        spaces = " " * random.randint(0, 5)
        print(f"{cyan}{spaces}{'>' * (i+1)}{dots} Neural Link {random.randint(1000, 9999)} established{reset}")
        time.sleep(0.2)
    print(f"{cyan}[ALL NEURAL PATHWAYS ACTIVE - ENHANCED SEARCH DISABLED]{reset}")
    print(f"{cyan}[MENTAT INTERFACE READY FOR SERVICE TO THE INFINITE EMPIRE]{reset}\n")
