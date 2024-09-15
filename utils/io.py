"""Contains functions used for io operations - read from files, write to files, etc"""

import logging
from pathlib import Path

logger = logging.getLogger()

def save_txt(txt: str, filepath: Path, encoding: str = "UTF8") -> None:
    """Save string as txt"""
    with open(filepath, "wt", encoding=encoding) as f:
        f.write(txt)
    logger.debug(f"Saved TXT {filepath}")