from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

class EmotionAnalyzer:
    def __init__(self):
        # Load pre-trained pipeline
        self.MODEL_NAME = "bhadresh-savani/distilbert-base-uncased-emotion"
        self.pipeline_classifier = pipeline("text-classification", model=self.MODEL_NAME, return_all_scores=True)

    def analyze_emotions(self, text):
        """
        Perform sentiment analysis for 6 emotions.
        Args:
            text (str): Input text to analyze.
        Returns:
            list: Emotion logits and scores.
        """
        return self.pipeline_classifier(text)
