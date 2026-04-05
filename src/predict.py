import joblib
import os
import sys
from typing import Dict

# ✅ Fix path (IMPORTANT)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import clean_text
from config import MODEL_PATH, VECTORIZER_PATH


class SentimentPredictor:
    def __init__(self):
        # ✅ Check if model exists
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError(
                "❌ Model or vectorizer not found. Please run train_model.py first."
            )

        # ✅ Load model
        self.model = joblib.load(MODEL_PATH)
        self.vectorizer = joblib.load(VECTORIZER_PATH)

    def predict(self, text: str) -> Dict:
        if not text or not isinstance(text, str):
            return {
                "label": "Invalid Input",
                "confidence": 0.0
            }

        # ✅ Clean text
        cleaned_text = clean_text(text)

        # ✅ Transform
        vector = self.vectorizer.transform([cleaned_text])

        # ✅ Predict
        prediction = self.model.predict(vector)[0]
        probabilities = self.model.predict_proba(vector)[0]

        confidence = max(probabilities)

        # ✅ Label mapping
        label_map = {
            0: "Negative 😡",
            1: "Positive 😊",
            2: "Neutral 😐"
        }

        return {
            "label": label_map.get(prediction, "Unknown"),
            "confidence": round(float(confidence) * 100, 2)
        }


# ✅ Test directly (optional)
if __name__ == "__main__":
    predictor = SentimentPredictor()

    while True:
        text = input("\nEnter review (or type 'exit'): ")
        if text.lower() == "exit":
            break

        result = predictor.predict(text)
        print("👉 Result:", result)