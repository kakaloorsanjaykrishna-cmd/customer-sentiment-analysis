from transformers import pipeline

class BERTSentiment:
    def __init__(self):
        self.classifier = pipeline("sentiment-analysis")

    def predict(self, text):
        result = self.classifier(text)[0]

        label = result['label']
        score = result['score']

        if label == "POSITIVE":
            sentiment = "Positive 😊"
        else:
            sentiment = "Negative 😡"

        return {
            "label": sentiment,
            "confidence": round(score * 100, 2)
        }