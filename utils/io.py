"""Contains functions used for io operations - read from files, write to files, etc"""

import logging
from pathlib import Path

from moviepy.video.io.VideoFileClip import VideoClip

logger = logging.getLogger()

def save_txt(txt: str, filepath: Path, encoding: str = "UTF8") -> None:
    """Save string as txt"""
    with open(filepath, "wt", encoding=encoding) as f:
        f.write(txt)
    logger.debug(f"Saved TXT {filepath}")


def save_video(video_clip: VideoClip, video_path: Path) -> None:
    """Save video to path"""
    video_path = str(video_path)
    video_clip.write_videofile(video_path, codec="libx264")
