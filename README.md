# Clip Extractor

Clip Extractor is a project inteded to extract sub clips from a video using face recognition. The script will recognize faces on which it was trained, and extract the clips where those faces appear

## Setup

### System libraries

some libs are needed as prerequisites:
- cmake
- ffmpeg

For example, in MacOS:
```sh
$ brew install cmake && brew install ffmpeg
```

### Python
#### Configure environment

Create and activate Python virtual environment:
```sh
$ python3 -m venv venv
$ source venv/bin/activate   # On macOS/Linux
# or
$ venv\Scripts\activate      # On Windows
```

Install the required Python libraries:
```sh
$ pip install -r requirements.txt
```

## Workflow (batch)

### 1.	Video Processing:

Process video {batch_size} frames at a time in order to set a limit on the amount of frames stored in memory at once:

1) Use OpenCV to read the first {batch_size} frames.
2) Perform face recognition to detetect the target face(s).
3) Record the frame indices where the face is recognized
4) Repeat until all video has been processed


### 2.	Extract Relevant Segments:
1) Use the timestamps from face detection to pinpoint relevant video segments.
2)	Use MoviePy to extract these segments and save them to the specified directory.

Extracted clips will have a standard length of {clips_length} frames. The script will extract {clips_length / 3} frames before the face was detected, and {2*clips_length / 3} frames after the face was detected.

If a face is detected more than once within the same range of frames, the clips for both detections will be merged

## Usage

python -m cli -l {log_dir} -q {quiet} batch -i {images_dir} -v {video_path} -f {frame_interval} -b {batch_size} -l {clips_length} -o {output_dir}

- log-dir (not required): Directory where to save logs. If None, logs are printed in stdout
- quiet (not required): if set to True, logging level is set to WARN, default is DEBUG
- images_dir: Directory with training face images
- video_path: Path to video to analyze
- frame_interval (not required): Frame interval to process. Default is 15 (process every 15th frame)
- clips_length (not reuqired): Length of output clips in frames
- output_dir: Directory where extracted clips are saved