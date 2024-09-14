"""Extract audio and frames from video"""

import os
from pathlib import Path
import cv2
import numpy as np

def extract_frames(video_path: Path, output_dir=None: Path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in the video: {total_frames}")

    # Create a directory to save frames if an output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    frame_list = []  # List to store frames if saving to memory
    frame_count = 0  # Frame counter

    # Read a frame from the video
    success, frame = cap.read()

    while success:

        # Optionally save the frame as an image file
        if output_dir:
            frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved {frame_filename}")

        # Optionally store the frame in memory as a NumPy array
        frame_list.append(frame)

        frame_count += 1

        success, frame = cap.read()

    # close video file
    cap.release()

    print(f"Extracted {frame_count} frames from the video.")

    # Return the list of frames if stored in memory
    return frame_list

