"""Contains functions to detect object types, manipulate objects, etc."""

import os
import logging

from datetime import datetime
from pathlib import Path
import click


logger = logging.getLogger()

def get_datetime():
    now = datetime.now()
    return now.strftime('%Y-%m-%d_%H-%M-%S') + f",{now.microsecond // 1000}"


def has_right_extension(filepath: Path, ext: str):
    """Check if a file has right extension"""
    if not filepath.is_file():
        return False
    return filepath.suffix == ext


def is_image_file(filepath: Path):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    if not filepath.is_file():
        return False
    return filepath.suffix in valid_extensions


def merge_overlapping_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Merges a list of overlapping or contiguous frame ranges into a list of non-overlapping ranges.

    The function takes a list of tuples where each tuple represents a start and end of a range.
    It then merges all the overlapping or contiguous ranges into the smallest possible number of 
    non-overlapping ranges.

    Args:
        ranges (list[tuple[int, int]]): A list of tuples where each tuple contains two integers
        representing the start and end of a range. The ranges must be sorted by the start value.

    Returns:
        list[tuple[int, int]]: A list of tuples representing the merged non-overlapping ranges.
    
    Example:
        >>> merge_overlapping_ranges([(1, 3), (2, 6), (8, 10), (9, 12)])
        [(1, 6), (8, 12)]
    
    Notes:
        - The input `ranges` must be sorted by the first element of the tuple (the start of the range).
        - If the input list is empty, an empty list will be returned.
        - Ranges are merged when one range's start is less than or equal to the previous range's end.
    """
    merged_ranges = []
    current_start, current_end = ranges[0]

    for start, end in ranges[1:]:
        if start <= current_end:  # Overlap
            current_end = max(current_end, end)
        else:
            merged_ranges.append((current_start, current_end))
            current_start, current_end = start, end

    # Append the last merged range
    merged_ranges.append((current_start, current_end))
    
    return merged_ranges


def validate_encodings_source(images_dir: Path, encodings_file: Path):
    if images_dir:
        if not os.path.isdir(images_dir):
            logger.critical(f"Images folder {images_dir} not found")
            return
    elif encodings_file:
        if not has_right_extension(encodings_file, ext=".npy"):
            logger.critical(f"Encodings file must be .npy - found: {encodings_file}")
    else:
        logger.fatal("No source of encodings provided")
        raise click.UsageError("You must provide either --images-dir or --encodings-file.")
