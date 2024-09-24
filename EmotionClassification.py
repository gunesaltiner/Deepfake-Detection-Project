from transformers import pipeline

class EmotionClassification:
    def __init__(self):
        self.classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")

    def classify_text(self, text):
        results = self.classifier(text)
        return results