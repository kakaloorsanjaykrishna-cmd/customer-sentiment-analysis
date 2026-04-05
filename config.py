from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "raw" / "reviews.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FEATURES = 5000