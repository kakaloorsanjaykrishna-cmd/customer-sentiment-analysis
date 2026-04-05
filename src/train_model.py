import pandas as pd
import joblib
import os
import sys
import matplotlib.pyplot as plt

# Fix imports (IMPORTANT)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import clean_text
from src.feature_extraction import build_vectorizer
from config import DATA_PATH, MODEL_PATH, VECTORIZER_PATH

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


def train():
    print("🔥 Training script started...")

    # ✅ Load dataset
    if not os.path.exists(DATA_PATH):
        print(f"❌ Dataset not found at: {DATA_PATH}")
        return

    print("📂 Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("📊 Dataset shape:", df.shape)
    print(df.head())

    # ✅ Check columns
    if "review" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("Dataset must contain 'review' and 'sentiment' columns")

    # ✅ Preprocess
    print("🧹 Cleaning text...")
    df["review"] = df["review"].astype(str).apply(clean_text)

    # ✅ Feature extraction
    print("🔢 Vectorizing text...")
    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(df["review"])
    y = df["sentiment"]

    # ✅ Split
    print("✂️ Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ✅ Train model
    print("🤖 Training model...")
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # ✅ Evaluate
    print("📊 Evaluating model...")
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)

    print("\n✅ Accuracy:", accuracy)
    print("\n📄 Classification Report:\n", classification_report(y_test, preds))

    # ✅ Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

    # ✅ Accuracy Graph
    plt.figure()
    plt.bar(["Accuracy"], [accuracy])
    plt.title("Model Accuracy")
    plt.show()

    # ✅ Save model
    print("💾 Saving model...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("🎉 Model + Vectorizer saved successfully!")


# 🔥 VERY IMPORTANT (this was your issue)
if __name__ == "__main__":
    train()