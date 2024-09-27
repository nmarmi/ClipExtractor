"""Handle face recognizing"""
import os
import logging
from pathlib import Path, PosixPath, WindowsPath
import face_recognition
import numpy as np

import utils as u

logger = logging.getLogger()

class NoKnownFaceEncodingsError(Exception):
    """Exception raised when no known face encodings are provided."""
    def __init__(self, message="No known face encodings were provided."):
        self.message = message
        super().__init__(self.message)


class FaceDetector():
    """Class that handles face recognition"""

    def __init__(self, known_faces: list[np.ndarray] = []):
        self.known_faces = known_faces

    def train_from_encodings(self, face_encodings: list[np.ndarray]) -> None:
        """add provided faces encoding to known face encodings"""
        self.known_faces.append(face_encodings)
        logger.info(f"Added {len(face_encodings)} encodings to model")

    def train_from_images(self, faces_dir: Path) -> None:
        """
        Extract face encodings from images directory and add to known face encodings
        
        Args:
            faces_dir (Path): directory with training images
        """
        logger.info(f"Extracting encodings from {faces_dir}")
        init_encodings = len(self.known_faces)
        if not faces_dir.is_dir():
            raise NotADirectoryError(f"path {faces_dir} is not a directory")

        # Loop through each image file in the specified directory
        for filename in os.listdir(faces_dir):
            # Load the image file
            image_path = Path(os.path.join(faces_dir, filename))
            
            if not u.is_image_file(image_path):
                continue
            
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                self.known_faces.append(encodings)
            
        logger.info(f"Extracted {len(self.known_faces) - init_encodings} encodings from {faces_dir}")

    def get_known_faces(self):
        """Returns known face encodings"""
        return self.known_faces

    def detect_faces(self, frame: np.ndarray):
        """
        Detect face locations and encodings in the current frame
        
        Args:
            frame (np.ndarray): frame to analyze stored in np.ndarray
        Returns:
            face_encodings (list[np.ndarray]): list of detected face encodings
        """
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        return face_encodings

    def known_face_detected(self, detected_face_encoding: np.ndarray) -> bool:
        """
        Compare known face encodings to a single face encoding detected in a frame

        Args:
            detected_face_encoding (np.ndarray): face encoding to compare
        Returns:
            bool: True iff any known face matched with detected face encoding
        """
        if not self.known_faces:
            raise NoKnownFaceEncodingsError()

        matches = face_recognition.compare_faces(self.known_faces, detected_face_encoding)
        
        for match in matches:
            if np.any(match):
                return True
        return False

    def get_timestamps(self, frame_list: list[np.ndarray], frame_interval: int) -> set[int]:
        """    
        Iterate through frames and save frames where known face is detected

        Args:
            frame_list (list[np.ndarrat]): List of frames (NumPy arrays) extracted from the video.
        
        Returns:
            set[int]: A set of timestamps (frame indices) where known faces were detected.
        """
        if not self.known_faces:
            raise NoKnownFaceEncodingsError()
        logger.info("Extracting timestamps")
        timestamps = set()
        for frame_index, frame in enumerate(frame_list):
            if frame_index%frame_interval == 0:
                face_encodings = self.detect_faces(frame)
                # compare each detected face encoding to known faces
                for face_encoding in face_encodings:
                    if self.known_face_detected(face_encoding):
                        timestamps.add(frame_index)

        return timestamps

#################################################################

    def execute_with_images(self, train_faces_dir: Path, frame_list: list[np.ndarray], frame_interval: int) -> set[int]:
        """
        Train model on faces from images directory, and identify frames with known faces
        Args:
            train_faces_dir (Path): Directory with training images
            frame_list (list[np.ndarrat]): List of frames (NumPy arrays) extracted from the video.
            frame_interval (int): frames interval to process
        
        Returns:
            set[int]: A set of timestamps (frame indices) where known faces were detected.
        """
        self.train_from_images(train_faces_dir)
        return self.get_timestamps(frame_list, frame_interval)


    def execute_pretrained(self, frame_list: list[np.ndarray], frame_interval: int) -> set[int]:
        """
        Execute pipleine with model pretrained on known faces, and identify frames with known faces
        Args:
            frame_list (list[np.ndarrat]): List of frames (NumPy arrays) extracted from the video.
            frame_interval (int): frames interval to process
        
        Returns:
            set[int]: A set of timestamps (frame indices) where known faces were detected.
        """
        return self.get_timestamps(frame_list, frame_interval)

    def execute_with_encodings(
            self,
            known_face_encodings: list[np.ndarray],
            frame_list: list[np.ndarray],
            frame_interval: int
        ) -> set[int]:
        """
        Add known face encodings to known faces, and identify frames with known faces
        Args:
            known_face_encodings (list[np.ndarray]): list of face encodigs
            frame_list (list[np.ndarrat]): List of frames (NumPy arrays) extracted from the video.
            frame_interval (int): frames interval to process
        
        Returns:
            set[int]: A set of timestamps (frame indices) where known faces were detected.
        """
        self.train_from_encodings(known_face_encodings)
        return self.get_timestamps(frame_list, frame_interval)

    def execute(self, *args, **kwargs) -> set[int]:
        """
        Execute the pipeline based on provided arguments.
        
        Arguments:
            - (images_dir: Path, frames: list[np.ndarray]): Directory with training images and frames list.
            - (known_encodings: list[np.ndarray], frames: list[np.ndarray]): Known face encodings and frames list.
            - (frames: list[np.ndarray],): List of frames (NumPy arrays) extracted from the video.
        
        Optional Keyword Arguments:
            - frame_interval (int): Frames interval to process. Default is 10 (process every 10th frame)

        Returns:
            set[int]: A list of timestamps (frame indices) where known faces were detected.
        """
        signature = tuple(
            Path if isinstance(arg, (PosixPath, WindowsPath)) else arg.__class__
            for arg in args
        )
        typemap = {
            (Path, list): self.execute_with_images,
            (list, list): self.execute_with_encodings,
            (list, ): self.execute_pretrained
        }
        if signature in typemap:
            frame_interval = kwargs.get('frame_interval', 10)
            return(typemap[signature](*args, frame_interval))
        else:
            raise TypeError(f"Invalid type signature: {signature}. Accepted signatures are: Path, list), (list, list), or (list)." 
                            "Optional keyword argument 'frame_interval' (int) is also supported.")

