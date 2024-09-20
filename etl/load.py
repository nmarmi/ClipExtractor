import logging
from pathlib import Path

from utils.io import save_video

import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip, VideoClip

logger = logging.getLogger()

def extract_video(video_path: Path, start: int, end: int) -> VideoClip:
    """
    Extract video from starting frame to ending frame

    Args:
        video_path (Path): path to video that was processed
        start (int): index of first frame
        end (int): index of last frame
    Returns:
        VideoClip: video object to write in file
    """
    video_path = str(video_path)
    video = VideoFileClip(video_path)
    fps = video.fps
    
    start_time = start / fps
    end_time = end / fps

    video_subclip = video.subclip(start_time, end_time)
    return video_subclip


def get_frame_ranges(frames_set: set[int], clip_length: int, video_length: int,) -> list[tuple[int, int]]:
    """
    Get list of (start, end) frame frames_set from set of frames. Used to later extract videos

    Args:
        frames_set (set[int]): set of frame indices
        clip_length (int): number of frames per clip
        video_length (int): total number of frames of input video
    
    Returns:
        list[tuple[int, int]]: list of (start, end) frame indices
    """
    ranges = [(max(i - clip_length/3, 0), min(i + int(2*clip_length/3), video_length)) for i in frames_set]
    ranges.sort()

    # Merge overlapping ranges
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


def process_extracted_frames(video_path: Path, frames: set, output_dir: Path, clips_length: int = 1800) -> None:
    """
    Given detected frames, extract subclips of original video

    Args:
        video_path (Path): path to original video
        frames (set): set of detected frames
        output_dir (Path): directory where to save subclips
        clips_length (int): number of frames per extracted subclips
    
    Returns:
        None
    """
    logger.info(f"Starting post processing for video {video_path.stem}")

    video_length = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT)
    frame_ranges = get_frame_ranges(frames, clips_length, video_length)
    i = 1
    logger.info(f"Saving extracted clips to {output_dir}")
    for range in frame_ranges:
        video_clip = extract_video(video_path, start=range[0], end=range[1])
        output_path = Path(output_dir) / f"extracted_clip_{i}.MP4"
        save_video(video_clip, output_path)
        i += 1
    logger.info(f"Saved {i-1} clips")