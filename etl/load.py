from pathlib import Path
from moviepy.video.io.VideoFileClip import VideoFileClip, VideoClip

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
    # Load the video file
    video = VideoFileClip(video_path)
    fps = video.fps
    
    start_time = start / fps
    end_time = end / fps

    video_subclip = video.subclip(start_time, end_time)
    return video_subclip


def get_frame_ranges(indices, num_frames=100) -> list[tuple[int, int]]:
    """
    Get list of (start, end) frame indices from set of frames. Used to later extract videos

    Args:
        indices (set[int]): set of frame indices
        num_frames (int): number of frames before and after detected frame to keep in video
    
    Returns:
        list[tuple[int, int]]: list of (start, end) frame indices
    """
    ranges = [(max(i - num_frames, 0), i + num_frames) for i in indices]
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
