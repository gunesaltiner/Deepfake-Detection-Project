import cv2
from PyQt5.QtWidgets import (
    QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget,
    QHBoxLayout, QFileDialog, QScrollArea, QDialog
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt

from Speech2Text import SpeechToText
from EmotionClassification import EmotionClassification

class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Camera App")
        self.setFixedSize(800, 750)

        # Create buttons
        self.open_camera_button = QPushButton("Open Camera", self)
        self.open_camera_button.clicked.connect(self.open_camera)

        self.upload_video_button = QPushButton("Upload Video", self)
        self.upload_video_button.clicked.connect(self.upload_video)

        # Create a label to display the video feed
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(800, 600)
        self.video_label.setAlignment(Qt.AlignCenter)

        # Layout for buttons
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.upload_video_button)
        button_layout.addWidget(self.open_camera_button)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.video_label)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Set up the timer for video feed updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.cap = None
        self.is_camera = False
        self.video_file = None

        # Initialize models (you might want to do this lazily to reduce startup time)
        self.speech_to_text = None
        self.emotion_classifier = None

    def open_camera(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(0)
        self.is_camera = True
        self.timer.start(20)
        # Implement camera recording if needed

    def upload_video(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "",
            "Video Files (*.avi *.mp4 *.mov *.mkv)", options=options
        )
        if file_name:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(file_name)
            self.is_camera = False
            self.timer.start(20)
            self.video_file = file_name

    def update_frame(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            step = channel * width
            q_img = QImage(frame.data, width, height, step, QImage.Format_RGB888)

            # Scale the QImage to fit the video_label size while maintaining aspect ratio
            scaled_q_img = q_img.scaled(self.video_label.size(), Qt.KeepAspectRatio)
            self.video_label.setPixmap(QPixmap.fromImage(scaled_q_img))
        else:
            # End of video reached
            self.timer.stop()
            self.cap.release()
            self.cap = None

            # Process the video file
            if self.video_file:
                self.process_video(self.video_file)
            else:
                # Handle the case when video_file is None
                pass

    def process_video(self, video_path):
        # Initialize SpeechToText if not already done
        if self.speech_to_text is None:
            self.speech_to_text = SpeechToText()

        # Extract audio and transcribe
        print(f"Converting {video_path} to MP3...")
        mp3_file = self.speech_to_text.video_to_mp3(video_path)
        print(f"Audio extracted and saved as {mp3_file}")

        # Transcribe the audio from the MP3 file
        print("Transcribing audio...")
        transcription = self.speech_to_text.transcribe(mp3_file)
        print("Transcription complete.")

        # Get the transcribed text
        transcribed_text = transcription["text"]

        # Save the transcription to a text file
        output_text_file = "transcription.txt"
        self.speech_to_text.save_transcription(transcribed_text, output_text_file)

        # Initialize EmotionClassification if not already done
        if self.emotion_classifier is None:
            self.emotion_classifier = EmotionClassification()

        # Perform emotion classification
        emotion_results = self.emotion_classifier.classify_text(transcribed_text)

        # Store the transcription and emotion results
        self.transcribed_text = transcribed_text
        self.emotion_results = emotion_results

        # Show the report window
        self.show_report_window()

    def show_report_window(self):
        report_window = QDialog(self)
        report_window.setWindowTitle("Video Report")
        report_window.setFixedSize(600, 400)

        # Create a label to display the text
        text_label = QLabel(report_window)
        text_label.setAlignment(Qt.AlignLeft)
        text_label.setWordWrap(True)
        text_label.setTextFormat(Qt.RichText)

        # Larger font for the text label
        font = QFont()
        font.setPointSize(14)
        text_label.setFont(font)

        # Prepare the display text
        display_text = f"<h2>Transcription:</h2><p>{self.transcribed_text}</p>"
        display_text += "<h2>Emotion Classification Results:</h2><ul>"

        for result in self.emotion_results:
            label = result['label']
            score = result['score']
            display_text += f"<li>{label}: {score:.4f}</li>"
        display_text += "</ul>"

        text_label.setText(display_text)

        # Make the text label scrollable
        scroll_area = QScrollArea(report_window)
        scroll_area.setWidget(text_label)
        scroll_area.setWidgetResizable(True)

        # Layout for the report window
        layout = QVBoxLayout()
        layout.addWidget(scroll_area)

        report_window.setLayout(layout)
        report_window.exec_()

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        event.accept()