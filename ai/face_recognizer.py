"""Handle face recognizing"""
import os
import face_recognition


def generate_face_encodings(images_dir) -> list:
    """
    Generates face encodings for a known face from images

    Args:
        images_dir (str): Directory containing images of known individuals.
    """
    face_encodings = []

    # Loop through each image file in the specified directory
    for filename in os.listdir(images_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Load the image file
            image_path = os.path.join(images_dir, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                print(f"Extracted encodings from {filename}")
                face_encodings.append(encodings)

    print(f"Found {len(face_encodings)} face encodings")
    return face_encodings


def process_frames_for_faces(frame_list, known_face_encodings):
    """
    Processes frames to detect faces and logs timestamps where specific faces are recognized.

    Args:
        frame_list (list): List of frames (NumPy arrays) extracted from the video.
        known_face_encodings (list): List of known face encodings to match against.
        
    Returns:
        list: A list of timestamps (frame indices) where known faces were detected.
    """
    detected_timestamps = set()

    for frame_index, frame in enumerate(frame_list):

        # Detect face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        # Check each detected face encoding against known faces
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            for match in matches:
                if any(match):
                    # If a known face is detected, save the timestamp (frame index)
                    detected_timestamps.add(frame_index)

    return detected_timestamps
