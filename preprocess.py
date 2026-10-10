"""Text preprocessing - train.py and app.py rendu-um ithaye use pannum,
so training-kum prediction-kum same cleaning irukkum."""
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

try:
    from nltk.stem import PorterStemmer
    _stemmer = PorterStemmer()
    stem = _stemmer.stem
except ImportError:           # nltk install aagala na stemming skip aagum
    def stem(word):
        return word

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
NUM_RE = re.compile(r"\d+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = URL_RE.sub(" urltoken ", text)          # links -> token
    text = re.sub(r"[£$€]", " currencytoken ", text)  # money symbols -> token
    text = NUM_RE.sub(" numtoken ", text)          # numbers -> token
    text = NON_ALPHA_RE.sub(" ", text)             # punctuation remove
    words = [stem(w) for w in text.split() if w not in ENGLISH_STOP_WORDS and len(w) > 1]
    return " ".join(words)
