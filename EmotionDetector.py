import cv2
from deepface import DeepFace
import tempfile
import os
import traceback

class EmotionDetector:
    def __init__(self):
        # Load the Haar Cascade face detection model
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def process_video(self, video_path):
        """
        Processes the video file at the given path and returns the dominant emotion detected.

        Parameters:
        - video_path (str): The file path to the video.

        Returns:
        - str: The dominant emotion detected in the video.
        """
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Dictionary to hold emotion counts
        emotions_count = {}

        while True:
            # Read frame from video
            ret, frame = cap.read()

            # If no frame is returned, end of video is reached
            if not ret:
                break

            # Convert frame to grayscale for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale frame
            faces = self.face_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            for (x, y, w, h) in faces:
                # Extract the face ROI (Region of Interest) from the original frame
                face_roi = frame[y:y + h, x:x + w]  # Original frame in BGR format

                # Check if face_roi is valid
                if face_roi.size == 0:
                    continue

                try:
                    # Save face_roi to a temporary file
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_image:
                        temp_filename = temp_image.name
                        cv2.imwrite(temp_filename, face_roi)

                    # Perform emotion analysis on the face ROI using the 'img_path' parameter
                    result = DeepFace.analyze(
                        img_path=temp_filename,
                        actions=['emotion'],
                        enforce_detection=False
                    )

                    # Access the dominant emotion from the result dictionary
                    emotion = result['dominant_emotion']

                    # Update emotions count
                    if emotion in emotions_count:
                        emotions_count[emotion] += 1
                    else:
                        emotions_count[emotion] = 1

                except Exception as e:
                    # Handle exceptions and print detailed debug information
                    print(f"An error occurred: {e}")
                    print(f"Type of face_roi: {type(face_roi)}")
                    print(f"Shape of face_roi: {face_roi.shape}")
                    traceback.print_exc()
                    continue
                finally:
                    # Delete the temporary file
                    if os.path.exists(temp_filename):
                        os.remove(temp_filename)

        # Release the video capture object
        cap.release()

        # Determine the overall dominant emotion
        if emotions_count:
            dominant_emotion = max(emotions_count, key=emotions_count.get)
            return dominant_emotion
        else:
            return "No emotion detected"
        
        
# Example usage:
detector = EmotionDetector()
dominant_emotion = detector.process_video('/Users/gunes/VisualStudioProjects/ArvisProject/video.mp4')
print(f"The dominant emotion in the video is: {dominant_emotion}")