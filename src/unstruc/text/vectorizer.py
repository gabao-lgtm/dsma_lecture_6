"""
Vectorizer — src/unstruc/text/vectorizer.py
============================================

A single class with a consistent sklearn-style interface that spans the
first two NLP eras:

  "bow"      — Bag-of-Words (CountVectorizer)         Era 1
  "tfidf"    — TF-IDF (TfidfVectorizer)               Era 1
  "word2vec" — Word2Vec mean-pooled dense vectors      Era 2

All three expose the same .fit() / .transform() / .fit_transform() contract
so the downstream LogisticRegression in the pipeline never needs to change —
only the representation layer swaps out.  This is the central architectural
lesson of the statistical and semantic eras.

BoW / TF-IDF return scipy sparse matrices (consistent with sklearn).
Word2Vec returns a dense numpy array.
Both are accepted by sklearn classifiers without modification.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec


class Vectorizer:
    """
    Unified text vectoriser for BoW, TF-IDF, and Word2Vec.

    Args:
        method  : one of "bow", "tfidf", "word2vec"
        **kwargs: forwarded to the underlying sklearn vectoriser or
                  Word2Vec constructor (e.g. max_features, vector_size)
    """

    def __init__(self, method: Literal["bow", "tfidf", "word2vec"], **kwargs):
        self.method = method
        self._vec   = None
        self._model = None

        if method == "bow":
            self._vec = CountVectorizer(**kwargs)
        elif method == "tfidf":
            self._vec = TfidfVectorizer(**kwargs)
        elif method == "word2vec":
            self._w2v_kwargs = kwargs
        else:
            raise ValueError(f"Unknown method '{method}'. Choose: bow, tfidf, word2vec")

    # ── Public interface ──────────────────────────────────────────────────────

    def fit(self, texts) -> "Vectorizer":
        texts = self._to_list(texts)
        if self.method in ("bow", "tfidf"):
            self._vec.fit(texts)
        else:
            tokenized = [t.split() for t in texts]
            self._model = Word2Vec(
                tokenized,
                vector_size = self._w2v_kwargs.get("vector_size", 100),
                window      = self._w2v_kwargs.get("window", 5),
                min_count   = self._w2v_kwargs.get("min_count", 2),
                workers     = self._w2v_kwargs.get("workers", 4),
                seed        = self._w2v_kwargs.get("seed", 42),
            )
        return self

    def transform(self, texts):
        texts = self._to_list(texts)
        if self.method in ("bow", "tfidf"):
            return self._vec.transform(texts)
        return self._mean_pool(texts)

    def fit_transform(self, texts):
        texts = self._to_list(texts)
        if self.method in ("bow", "tfidf"):
            return self._vec.fit_transform(texts)
        return self.fit(texts).transform(texts)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        if self.method in ("bow", "tfidf"):
            return len(self._vec.vocabulary_)
        return len(self._model.wv)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _mean_pool(self, texts) -> np.ndarray:
        size = self._model.vector_size
        result = []
        for text in texts:
            vecs = [
                self._model.wv[w]
                for w in text.split()
                if w in self._model.wv
            ]
            result.append(np.mean(vecs, axis=0) if vecs else np.zeros(size))
        return np.array(result)

    @staticmethod
    def _to_list(texts) -> list:
        if isinstance(texts, (pd.Series, np.ndarray)):
            return texts.tolist()
        return list(texts)
