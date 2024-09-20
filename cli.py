"""Script Orchestrator"""
import logging
from pathlib import Path
import click
from ai.face_recognizer import FaceDetector
from etl.extract import extract_frames
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
    "-l",
    "--clips-length",
    required=False,
    default=1800,
    type=int,
    help="length of output clips in frames",
)
@click.option(
    "-o",
    "--output-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Output file",
)
@click.pass_context
def run(ctx: click.core.Context, images_dir: Path, video_path: Path, frame_interval: int, clips_length: int, output_dir: Path):
    logger = ctx.obj["logger"]
    logger.info("etracring frames from video")
    frames = extract_frames(video_path)
    
    face_detector = FaceDetector()
    timestamps = face_detector.execute(images_dir, frames, frame_interval=frame_interval)
    process_extracted_frames(video_path, timestamps, output_dir, clips_length=clips_length)
    

if __name__ == "__main__":
    main()
