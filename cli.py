"""Script Orchestrator"""
import logging
from pathlib import Path
import click
import os

from ai.face_recognizer import FaceDetector
from etl.extract import extract_frames, extract_batch_frames, get_total_frames
from etl.load import process_extracted_frames

import utils as u

@click.group()
@click.option(
    "-l",
    "--log-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory where to save logs. If None, logs are printed in stdout",
)
@click.option(
    "-q",
    "--quiet",
    default=False,
    is_flag=True,
    help="Set logging level to WARN, default is DEBUG",
)
@click.pass_context
def main(
    ctx: click.core.Context, log_dir: Path | None, quiet: bool
):
    """CLI endpoint. Orchestrate all commands."""
    
    #initialize logger
    logger = u.config_logger(
        log_dir=log_dir, level=logging.INFO if quiet else logging.DEBUG
    )
    logger.info("Starting main.")

    ctx.obj = {
        "logger": logger
    }


@main.command()
@click.option(
    "-i",
    "--images-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory with training face images",
)
@click.option(
    "-v",
    "--video-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to video to analyze",
)
@click.option(
    "-f",
    "--frame-interval",
    required=False,
    default=10,
    type=int,
    help="Frame interval to process. Default is 10 (process every 10th frame)",
)
@click.option(
    "-o",
    "--output-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Output file",
)
@click.pass_context
def detect_faces(ctx: click.core.Context, images_dir: Path, video_path: Path, frame_interval: int, output_path: Path):
    #extract logger
    logger = ctx.obj["logger"]
    
    logger.info("Starting detect faces")
    frames = extract_frames(video_path)
    
    face_detector = FaceDetector()
    timestamps = face_detector.execute(images_dir, frames, frame_interval=frame_interval)
    timestamps = sorted(timestamps)
    u.save_txt(str(timestamps), output_path)


@main.command()
@click.option(
    "-i",
    "--images-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory with training face images",
)
@click.option(
    "-o",
    "--output-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Output file",
)
@click.pass_context
def generate_encodings(ctx: click.core.Context, images_dir: Path, output_path: Path):
    #extract logger
    logger = ctx.obj["logger"]
    
    # validation steps
    if not os.path.isdir(images_dir):
            logger.critical(f"Images folder {images_dir} not found")
            return
    
    if not u.has_right_extension(output_path, ext=".npy"):
            logger.critical(f"Encodings file must be .npy - found: {output_path}")
            return

    logger.info("Starting generate encodings")
    face_detector = FaceDetector()
    face_detector.train_from_images(images_dir)
    encodings = face_detector.get_known_faces()
    u.save_encodings(encodings, output_path)


@main.command()
@click.option(
    "-i",
    "--images-dir",
    required=False,  # Not required if encodings-file is provided
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory with training face images",
)
@click.option(
    "-e",
    "--encodings-file",
    required=False,  # Not required if images-dir is provided
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="File with pre-saved face encodings",
)
@click.option(
    "-v",
    "--video-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to video to analyze",
)
@click.option(
    "-f",
    "--frame-interval",
    required=False,
    default=15,
    type=int,
    help="Frame interval to process. Default is 15 (process every 15th frame)",
)
@click.option(
    "-l",
    "--clips-length",
    required=False,
    default=1800,
    type=int,
    help="Length of output clips in frames",
)
@click.option(
    "-o",
    "--output-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory where extracted clips are saved",
)
@click.pass_context
def run(
    ctx: click.core.Context,
    images_dir: Path,
    encodings_file: Path,
    video_path: Path,
    frame_interval: int,
    clips_length: int,
    output_dir: Path
):
    logger = ctx.obj["logger"]
    
    # validation steps
    u.validate_encodings_source(images_dir, encodings_file)

    if not Path(video_path).exists:
        logger.critical(f"Input video {video_path} not found")

    if not os.path.isdir(output_dir):
        logger.critical(f"Output folder {output_dir} not found")

    logger.info("Extracting frames from video")
    frames = extract_frames(video_path)
    
    face_detector = FaceDetector()
    if encodings_file:
        encodings = u.load_encodings(encodings_file)
        face_detector.train_from_encodings(encodings)
    else:
        face_detector.train_from_images(images_dir)
    
    if len(face_detector.get_known_faces()) == 0:
        raise ValueError("No face encodings found")

    timestamps = face_detector.execute(frames, frame_interval=frame_interval)
    process_extracted_frames(video_path, timestamps, output_dir, clips_length=clips_length)


@main.command()
@click.option(
    "-i",
    "--images-dir",
    required=False,  # Not required if encodings-file is provided
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory with training face images",
)
@click.option(
    "-e",
    "--encodings-file",
    required=False,  # Not required if images-dir is provided
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="File with pre-saved face encodings",
)
@click.option(
    "-v",
    "--video-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to video to analyze",
)
@click.option(
    "-f",
    "--frame-interval",
    required=False,
    default=15,
    type=int,
    help="Frame interval to process. Default is 15 (process every 15th frame)",
)
@click.option(
    "-b",
    "--batch-size",
    required=False,
    default=2000,
    type=int,
    help="Number of frames to be processed in each batch",
)
@click.option(
    "-l",
    "--clips-length",
    required=False,
    default=1800,
    type=int,
    help="Length of output clips in frames",
)
@click.option(
    "-o",
    "--output-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Output file",
)
@click.pass_context
def batch(
    ctx: click.core.Context,
    images_dir: Path,
    encodings_file: Path,
    video_path: Path,
    frame_interval: int,
    batch_size: int,
    clips_length: int,
    output_dir: Path
):
    logger = ctx.obj["logger"]
    
    # validation steps
    u.validate_encodings_source(images_dir, encodings_file)
    
    if not Path(video_path).exists:
        logger.critical(f"Input video {video_path} not found")

    if not os.path.isdir(output_dir):
        logger.critical(f"Output folder {output_dir} not found")
    

    logger.info("Extracting frames from video")
    current_frame = 0
    total_frames = get_total_frames(video_path)
    logger.debug(f"Total frames in video: {total_frames}")
    #initialize face detector
    face_detector = FaceDetector()
    if encodings_file:
        encodings = u.load_encodings(encodings_file)
        face_detector.train_from_encodings(encodings)
    else:
        face_detector.train_from_images(images_dir)

    timestamp_lists = []
    batch_count = 1
    logger.info("Starting batch processing")
    while current_frame < total_frames:
        frames = extract_batch_frames(video_path, current_frame, batch_size, total_frames)

        timestamp_lists.append(face_detector.execute(frames, frame_interval=frame_interval))
        current_frame += batch_size
        logger.debug(f"Processed batch {batch_count}")
        batch_count += 1
        # clear memory
        frames = []

    process_extracted_frames(video_path, timestamp_lists, output_dir, clips_length=clips_length)
    

if __name__ == "__main__":
    main()
