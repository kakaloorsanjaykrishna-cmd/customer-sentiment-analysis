import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

STOPWORDS = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()

    words = [w for w in words if w not in STOPWORDS and len(w) > 2]

    return " ".join(words)