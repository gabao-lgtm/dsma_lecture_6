"""
Transformer Encoder (BERT) — src/unstruc/text/transformer_encoder.py
=====================================================================

Era 4 of the NLP evolution story: the Attention mechanism.

Transformers discard recurrence entirely.  Self-attention lets every token
attend to every other token in the sequence simultaneously, making the
architecture fully parallelisable and enabling training on internet-scale
corpora.

We use bert-base-uncased with a single linear classification head added on
top of the [CLS] pooled output.  Fine-tuning for 2 epochs on a small subset
is enough to demonstrate the accuracy gap vs. LSTM.

The extract_embeddings() function returns [CLS] token vectors — contextual
sentence representations used downstream for embedding drift detection.

Public API
----------
  get_tokenizer()                               → BertTokenizer
  BertClassifier(nn.Module)                     → model class
  train_bert(X_train, y_train, ...)             → (BertClassifier, tokenizer)
  predict_bert(model, tokenizer, texts, ...)    → np.ndarray of binary labels
  extract_embeddings(model, tokenizer, texts, ...) → np.ndarray shape (N, 768)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer

MODEL_NAME = "bert-base-uncased"


# ── Dataset ───────────────────────────────────────────────────────────────────

class _BertDataset(Dataset):
    def __init__(self, encodings: dict, labels: list[int] = None):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


# ── Model ─────────────────────────────────────────────────────────────────────

class BertClassifier(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.bert       = BertModel.from_pretrained(MODEL_NAME)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
        )
        logits = self.classifier(self.dropout(outputs.pooler_output)).squeeze(-1)
        if labels is not None:
            loss = nn.BCEWithLogitsLoss()(logits, labels)
            return loss, logits
        return logits


# ── Tokenisation helper ───────────────────────────────────────────────────────

def get_tokenizer() -> BertTokenizer:
    return BertTokenizer.from_pretrained(MODEL_NAME)


def _encode(texts, tokenizer: BertTokenizer, max_len: int) -> dict:
    return tokenizer(
        list(texts),
        padding      = True,
        truncation   = True,
        max_length   = max_len,
        return_tensors = None,
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train_bert(
    X_train_texts,
    y_train,
    epochs:     int   = 2,
    batch_size: int   = 16,
    lr:         float = 2e-5,
    max_len:    int   = 128,
    device:     str   = None,
) -> tuple[BertClassifier, BertTokenizer]:
    """
    Fine-tune bert-base-uncased on the provided review texts.

    Two epochs on 2 000 samples runs in roughly 5–10 minutes on CPU,
    making it feasible for a classroom demo.  A GPU will cut this to ~1 min.

    Args:
        X_train_texts : list / Series of lightly preprocessed review strings
                        (preprocessor mode="minimal" — BERT handles its own tokenisation)
        y_train       : binary labels (0 / 1)
        epochs        : fine-tuning epochs
        batch_size    : mini-batch size (16 is safe for 8 GB RAM)
        lr            : AdamW learning rate (2e-5 is standard for BERT fine-tuning)
        max_len       : maximum token sequence length (128 keeps memory low)
        device        : "cuda" / "cpu" (auto-detected if None)

    Returns:
        Tuple of (trained BertClassifier, BertTokenizer).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer  = get_tokenizer()
    encodings  = _encode(X_train_texts, tokenizer, max_len)
    labels     = list(y_train) if not isinstance(y_train, list) else y_train
    dataset    = _BertDataset(encodings, labels)
    loader     = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model     = BertClassifier().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            labels_batch = batch.pop("labels").to(device)
            batch        = {k: v.to(device) for k, v in batch.items()}
            optimiser.zero_grad()
            loss, _ = model(**batch, labels=labels_batch)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
        print(f"    Epoch {epoch + 1}/{epochs}  loss={total_loss / len(loader):.4f}")

    return model, tokenizer


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_bert(
    model:      BertClassifier,
    tokenizer:  BertTokenizer,
    texts,
    batch_size: int = 32,
    max_len:    int = 128,
    device:     str = None,
) -> np.ndarray:
    """Return binary predictions (0 / 1) for a list of review strings."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    encodings = _encode(texts, tokenizer, max_len)
    dataset   = _BertDataset(encodings)
    loader    = DataLoader(dataset, batch_size=batch_size)

    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch  = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch)
            preds.extend((torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int))

    return np.array(preds)


# ── Embedding extraction ──────────────────────────────────────────────────────

def extract_embeddings(
    model:      BertClassifier,
    tokenizer:  BertTokenizer,
    texts,
    batch_size: int = 32,
    max_len:    int = 128,
    device:     str = None,
) -> np.ndarray:
    """
    Extract [CLS] pooled embeddings from bert-base-uncased.

    Returns a (N, 768) array of contextual sentence representations.
    These are passed to run_embedding_drift_report() in
    drift_detection_evidently.py to detect distribution shift in the
    embedding space between reference and current data.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    encodings = _encode(texts, tokenizer, max_len)
    dataset   = _BertDataset(encodings)
    loader    = DataLoader(dataset, batch_size=batch_size)

    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            batch  = {k: v.to(device) for k, v in batch.items()}
            output = model.bert(**batch)
            embeddings.append(output.pooler_output.cpu().numpy())

    return np.vstack(embeddings)
