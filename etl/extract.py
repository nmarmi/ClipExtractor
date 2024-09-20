"""Extract audio and frames from video"""
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger()

def extract_frames(video_path: Path) -> list[np.ndarray]:
    """Extract frames from video
    
    Args:
        video_path (Path): path of video to process

    Returns:
        list[np.ndarray]: list of all video's frames
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Total frames in the video: {total_frames}")

    frame_list = []  # List to store frames if saving to memory
    frame_count = 0  # Frame counter

    # Read first frame from the video
    success, frame = cap.read()

    while success:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Store the frame in memory as a NumPy array
        frame_list.append(frame)
        frame_count += 1
        success, frame = cap.read()

    # close video file
    cap.release()

    logger.info(f"Extracted {frame_count} frames from the video.")

    # Return the list of frames if stored in memory
    return frame_list


def extract_batch_frames(video_path: Path, start_frame: int, batch_size: int, total_frames: int) -> list[np.ndarray]:
    """Extract frames from video
    
    Args:
        video_path (Path): path of video to process
        start_frame (int): first frame of batch
        batch_size (int): number of frames to process at a time

    Returns:
        list[np.ndarray]: list of all batch's frames
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Set the video capture to start from the specific frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames_list = []
    
    current_frame = start_frame

    # Extract frames from start_frame to start_frame + batch_size (or until the end of the video)
    while current_frame < (start_frame + batch_size) and current_frame < total_frames:
        ret, frame = cap.read()

        if not ret:
            logger.error(f"Error reading frame at {current_frame}")
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frames_list.append(frame)
        current_frame += 1

    # Release video capture object
    cap.release()

    logger.info(f"Extracted {len(frames_list)} frames starting from frame {start_frame}")
    return frames_list


def get_total_frames(video_path):
    """Returns the total number of frames in the video."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()  # Release the video file resource
    return total_frames
