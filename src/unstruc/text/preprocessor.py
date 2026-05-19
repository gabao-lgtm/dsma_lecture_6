"""
Text Preprocessor — src/unstruc/text/preprocessor.py
======================================================

Two preprocessing modes to serve different NLP eras:

  "full"    — clean → tokenize → remove stopwords → lemmatize → rejoin
              Used for statistical (BoW / TF-IDF), semantic (Word2Vec), and
              sequential (LSTM) eras, where we build our own vocabulary.

  "minimal" — clean only (HTML, URLs, extra whitespace)
              Used for BERT and other pretrained models whose own tokenizer
              handles subword splitting; aggressive preprocessing would destroy
              the token distributions the model was pretrained on.

Public API
----------
  preprocess(text, mode)          → cleaned string
  preprocess_series(texts, mode)  → pd.Series of cleaned strings
"""

import re
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download("punkt",        quiet=True)
nltk.download("punkt_tab",    quiet=True)
nltk.download("stopwords",    quiet=True)
nltk.download("wordnet",      quiet=True)

_STOP_WORDS  = set(stopwords.words("english"))
_LEMMATIZER  = WordNetLemmatizer()
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<[^>]+>",  " ", text)   # HTML tags
    text = re.sub(r"http\S+",  " ", text)   # URLs
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\d+",      " ", text)
    text = re.sub(r"\s+",      " ", text).strip()
    return text


def preprocess(text: str, mode: str = "full") -> str:
    """
    Clean and optionally tokenize + normalise a single review string.

    Args:
        text : raw review text
        mode : "full" (BoW / W2V / LSTM) or "minimal" (BERT / LLM)

    Returns:
        Preprocessed string ready for vectorisation.
    """
    if mode == "minimal":
        return _clean(text)

    tokens = word_tokenize(_clean(text))
    tokens = [t for t in tokens if t not in _STOP_WORDS]
    tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def preprocess_series(texts: pd.Series, mode: str = "full") -> pd.Series:
    """Apply preprocess() to every element of a pandas Series."""
    return texts.fillna("").apply(lambda t: preprocess(t, mode=mode))
