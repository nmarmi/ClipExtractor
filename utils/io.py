"""Contains functions used for io operations - read from files, write to files, etc"""

import logging
from pathlib import Path
import numpy as np

from moviepy.video.io.VideoFileClip import VideoClip

logger = logging.getLogger()

def load_encodings(encodings_file_path: Path) -> list[np.ndarray]:
    logger.info(f"Reading encodings from {encodings_file_path}")
    with open(encodings_file_path, "rb") as f:
        known_face_encodings = np.load(f)
    return known_face_encodings

def save_encodings(encodings: list[np.ndarray], file_path: Path) -> None:
    """
    Saves a list of face encodings (np.ndarray) to a .npy file.

    Args:
        encodings (list[np.ndarray]): List of face encodings to save.
        file_path (Path): Path to the output .npy file.
    """
    np.save(file_path, encodings)
    logger.debug(f"Saved encodings to {file_path}")

def save_txt(txt: str, filepath: Path, encoding: str = "UTF8") -> None:
    """Save string as txt"""
    with open(filepath, "wt", encoding=encoding) as f:
        f.write(txt)
    logger.debug(f"Saved TXT {filepath}")


def save_video(video_clip: VideoClip, video_path: Path) -> None:
    """Save video to path"""
    video_path = str(video_path)
    video_clip.write_videofile(video_path, codec="libx264", logger=None)
