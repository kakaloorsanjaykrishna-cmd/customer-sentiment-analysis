from sklearn.feature_extraction.text import TfidfVectorizer
from config import MAX_FEATURES

def build_vectorizer():
    return TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),
        stop_words="english"
    )