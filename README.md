
# Video-to-Text-to-Emotion

**Video-to-Text-to-Emotion** is a Python-based GUI application designed to upload, display, and analyze videos. It uses advanced models to transcribe speech to text and classify emotions in real-time.

## Features

- **Upload Video**: Users can upload a video file using the GUI.
- **Video Playback**: The uploaded video is displayed within the application.
- **Speech-to-Text**: Once the video finishes playing, it transcribes the spoken content into text.
- **Emotion Detection**: The application detects and reports emotions based on the transcribed text.

## Project Structure

- **Video-to-Text-to-Emotion.py**: Main GUI application script. Handles video upload, playback, transcription, and emotion reporting.
- **Speech2Text.py**: Contains functions for converting speech to text using a pre-trained model.
- **EmotionClassification.py**: Analyzes the transcribed text to classify emotions.
- **main.py**: Entry point for running the application.

## Prerequisites

- Python 3.8 or higher
- Required libraries (listed in `environment.yml`)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/gunesaltiner/Video-to-Text-to-Emotion.git
    cd Video-to-Text-to-Emotion
    ```

2. Create and activate the conda environment:

    ```bash
    conda env create -f environment.yml
    conda activate video-to-text-to-emotion
    ```

3. Run the application:

    ```bash
    python main.py
    ```

## Usage

1. **Upload Video**: Click the 'Upload Video' button to select a video file.
2. **Play Video**: The uploaded video will be displayed in the application window.
3. **Transcription & Emotion Detection**: Once the video finishes playing, the application will transcribe the speech and detect emotions, displaying the results in the GUI.

## Contributing

Feel free to submit issues, fork the repository, and send pull requests!
