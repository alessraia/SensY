import random
import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# =====================
# Utility functions
# =====================

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_device(device=None):
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _batch_loader(X_tensor, y_tensor, batch_size: int, shuffle: bool):
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _binary_predict_from_probs(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (probs >= threshold).astype(int)


# =====================
# MLP on feature vectors
# =====================

class _MLPNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(256, 64), dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


class TorchMLPClassifier:
    def __init__(
        self,
        hidden_dims=(256, 64),
        dropout=0.3,
        lr=1e-3,
        batch_size=64,
        epochs=10,
        threshold=0.5,
        random_state=42,
        device=None,
        verbose=False,
    ):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.threshold = threshold
        self.random_state = random_state
        self.device = _get_device(device)
        self.verbose = verbose
        self.classes_ = np.array([0, 1])
        self.model_ = None
        self.feature_mean_ = None
        self.feature_std_ = None

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.feature_mean_) / self.feature_std_

    def fit(self, X, y):
        _set_seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        self.feature_mean_ = X.mean(axis=0, keepdims=True)
        self.feature_std_ = X.std(axis=0, keepdims=True) + 1e-8
        X = self._standardize(X)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        loader = _batch_loader(X_tensor, y_tensor, batch_size=self.batch_size, shuffle=True)

        self.model_ = _MLPNet(
            input_dim=X.shape[1],
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)

        pos_count = max(float((y == 1).sum()), 1.0)
        neg_count = max(float((y == 0).sum()), 1.0)
        pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        self.model_.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                logits = self.model_(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * xb.size(0)

            if self.verbose:
                avg_loss = epoch_loss / len(loader.dataset)
                print(f"[MLP] epoch {epoch + 1}/{self.epochs} - loss={avg_loss:.4f}")

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        X = self._standardize(X)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_tensor)
            probs_pos = torch.sigmoid(logits).cpu().numpy()

        probs_neg = 1.0 - probs_pos
        return np.column_stack([probs_neg, probs_pos])

    def predict(self, X):
        probs_pos = self.predict_proba(X)[:, 1]
        return _binary_predict_from_probs(probs_pos, threshold=self.threshold)


# =====================
# TextCNN on raw text
# =====================

class _TextCNNNet(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_filters: int, kernel_sizes, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.transpose(1, 2)
        conv_outs = [torch.relu(conv(emb)) for conv in self.convs]
        pooled = [torch.max(c, dim=2).values for c in conv_outs]
        features = torch.cat(pooled, dim=1)
        features = self.dropout(features)
        return self.fc(features).squeeze(1)


class TorchTextCNNClassifier:
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(
        self,
        vocab_size=20000,
        max_len=64,
        embed_dim=128,
        num_filters=100,
        kernel_sizes=(3, 4, 5),
        dropout=0.3,
        lr=1e-3,
        batch_size=64,
        epochs=8,
        threshold=0.5,
        min_freq=2,
        random_state=42,
        device=None,
        verbose=False,
    ):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.threshold = threshold
        self.min_freq = min_freq
        self.random_state = random_state
        self.device = _get_device(device)
        self.verbose = verbose
        self.classes_ = np.array([0, 1])
        self.word2idx_ = None
        self.model_ = None

    def _tokenize(self, text: str):
        return re.findall(r"\b\w+\b", str(text).lower())

    def _build_vocab(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(self._tokenize(text))

        most_common = [
            token for token, freq in counter.most_common()
            if freq >= self.min_freq
        ]
        most_common = most_common[: max(self.vocab_size - 2, 0)]

        self.word2idx_ = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
        }
        for idx, token in enumerate(most_common, start=2):
            self.word2idx_[token] = idx

    def _encode_text(self, text: str):
        tokens = self._tokenize(text)
        ids = [self.word2idx_.get(tok, 1) for tok in tokens[: self.max_len]]
        if len(ids) < self.max_len:
            ids.extend([0] * (self.max_len - len(ids)))
        return ids

    def _vectorize(self, texts):
        encoded = [self._encode_text(text) for text in texts]
        return np.asarray(encoded, dtype=np.int64)

    def fit(self, X, y):
        _set_seed(self.random_state)

        X = np.asarray(X, dtype=object)
        y = np.asarray(y, dtype=np.float32)

        self._build_vocab(X)
        X_encoded = self._vectorize(X)

        X_tensor = torch.tensor(X_encoded, dtype=torch.long)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        loader = _batch_loader(X_tensor, y_tensor, batch_size=self.batch_size, shuffle=True)

        self.model_ = _TextCNNNet(
            vocab_size=len(self.word2idx_),
            embed_dim=self.embed_dim,
            num_filters=self.num_filters,
            kernel_sizes=self.kernel_sizes,
            dropout=self.dropout,
        ).to(self.device)

        pos_count = max(float((y == 1).sum()), 1.0)
        neg_count = max(float((y == 0).sum()), 1.0)
        pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        self.model_.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                logits = self.model_(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * xb.size(0)

            if self.verbose:
                avg_loss = epoch_loss / len(loader.dataset)
                print(f"[TextCNN] epoch {epoch + 1}/{self.epochs} - loss={avg_loss:.4f}")

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=object)
        X_encoded = self._vectorize(X)
        X_tensor = torch.tensor(X_encoded, dtype=torch.long).to(self.device)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_tensor)
            probs_pos = torch.sigmoid(logits).cpu().numpy()

        probs_neg = 1.0 - probs_pos
        return np.column_stack([probs_neg, probs_pos])

    def predict(self, X):
        probs_pos = self.predict_proba(X)[:, 1]
        return _binary_predict_from_probs(probs_pos, threshold=self.threshold)