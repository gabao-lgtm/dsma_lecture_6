"""
Sequence Model (LSTM) — src/unstruc/text/sequence_model.py
===========================================================

Era 3 of the NLP evolution story: sequential modelling with LSTMs.

LSTMs read text left-to-right, maintaining a hidden state that acts as
compressed memory of everything seen so far.  They fix the vanishing-gradient
problem of vanilla RNNs via gating mechanisms, enabling longer-range
dependencies.

Breaking point: token N cannot be computed until token N-1 is done.
This sequential dependency fundamentally prevents parallelisation on
modern GPUs and is the direct motivation for the Transformer.

Public API
----------
  build_vocab(texts, max_vocab)               → word-to-index dict
  texts_to_sequences(texts, vocab, max_len)   → list of int lists
  train_lstm(X_train, y_train, vocab, ...)    → trained LSTMClassifier
  predict_lstm(model, texts, vocab, ...)      → np.ndarray of binary labels
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


# ── Vocabulary ────────────────────────────────────────────────────────────────

def build_vocab(texts, max_vocab: int = 20_000) -> dict:
    """Build a word → integer index mapping from a collection of texts."""
    counts = Counter(word for text in texts for word in str(text).split())
    vocab  = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counts.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab


def texts_to_sequences(texts, vocab: dict, max_len: int = 200) -> list[list[int]]:
    """Encode each text as a list of vocabulary indices, truncated to max_len."""
    unk = vocab["<UNK>"]
    return [
        [vocab.get(w, unk) for w in str(t).split()][:max_len]
        for t in texts
    ]


# ── Dataset ───────────────────────────────────────────────────────────────────

class _ReviewDataset(Dataset):
    def __init__(self, sequences: list[list[int]], labels: list[int]):
        self.seqs   = sequences
        self.labels = labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq   = torch.tensor(self.seqs[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return seq, label


def _collate(batch):
    seqs, labels = zip(*batch)
    padded = pad_sequence(seqs, batch_first=True, padding_value=0)
    return padded, torch.stack(labels)


# ── Model ─────────────────────────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 hidden_dim: int = 128, n_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(
            embed_dim, hidden_dim, n_layers,
            batch_first = True,
            dropout     = dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb              = self.dropout(self.embedding(x))
        _, (hidden, _)   = self.lstm(emb)
        return self.fc(self.dropout(hidden[-1])).squeeze(1)


# ── Training ──────────────────────────────────────────────────────────────────

def train_lstm(
    X_train_texts,
    y_train,
    vocab:      dict,
    epochs:     int   = 3,
    batch_size: int   = 64,
    lr:         float = 1e-3,
    device:     str   = None,
) -> LSTMClassifier:
    """
    Train an LSTMClassifier on tokenised review texts.

    Args:
        X_train_texts : list / Series of preprocessed review strings
        y_train       : list / Series / array of binary labels (0 or 1)
        vocab         : output of build_vocab()
        epochs        : training epochs (3 is enough for a classroom demo)
        batch_size    : mini-batch size
        lr            : Adam learning rate
        device        : "cuda" / "cpu" (auto-detected if None)

    Returns:
        Trained LSTMClassifier.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    seqs    = texts_to_sequences(X_train_texts, vocab)
    labels  = list(y_train) if not isinstance(y_train, list) else y_train
    dataset = _ReviewDataset(seqs, labels)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         collate_fn=_collate)

    model     = LSTMClassifier(vocab_size=len(vocab)).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for seqs_batch, labels_batch in loader:
            seqs_batch   = seqs_batch.to(device)
            labels_batch = labels_batch.to(device)
            optimiser.zero_grad()
            loss = criterion(model(seqs_batch), labels_batch)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
        print(f"    Epoch {epoch + 1}/{epochs}  loss={total_loss / len(loader):.4f}")

    return model


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_lstm(
    model:      LSTMClassifier,
    texts,
    vocab:      dict,
    batch_size: int = 128,
    device:     str = None,
) -> np.ndarray:
    """Return binary predictions (0 / 1) for a list of review strings."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    seqs    = texts_to_sequences(texts, vocab)
    dataset = _ReviewDataset(seqs, [0] * len(seqs))
    loader  = DataLoader(dataset, batch_size=batch_size, collate_fn=_collate)

    model.eval()
    preds = []
    with torch.no_grad():
        for seqs_batch, _ in loader:
            seqs_batch = seqs_batch.to(device)
            logits     = model(seqs_batch)
            preds.extend((torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int))

    return np.array(preds)
