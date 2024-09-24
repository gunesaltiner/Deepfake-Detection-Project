import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from moviepy.editor import VideoFileClip

class SpeechToText:
    def __init__(self, model_id="openai/whisper-large-v3"):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device
        )

    def video_to_mp3(self, video_path, output_audio_path="output_audio.mp3"):
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(output_audio_path, codec="mp3")
        audio_clip.close()
        video_clip.close()
        return output_audio_path

    def transcribe(self, audio_input):
        return self.pipe(audio_input, generate_kwargs={"language": "english"})
    
    def save_transcription(self, text, file_path):
        with open(file_path, "w") as text_file:
            text_file.write(text)
        print(f"Transcription saved to {file_path}")