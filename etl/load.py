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

