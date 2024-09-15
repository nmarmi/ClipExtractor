"""Extract audio and frames from video"""
import logging
from pathlib import Path
import cv2

logger = logging.getLogger()

def extract_frames(video_path: Path):
    """extract frames from video"""
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
