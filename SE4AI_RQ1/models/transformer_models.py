import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


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


class TorchDistilBERTClassifier:
    """
    Sklearn-like wrapper around DistilBERT fine-tuning for binary text classification.
    Expects raw text input (question_en).
    """

    def __init__(
        self,
        pretrained_model_name="distilbert-base-uncased",
        max_len=128,
        lr=2e-5,
        batch_size=16,
        epochs=3,
        threshold=0.5,
        random_state=42,
        device=None,
        verbose=False,
    ):
        self.pretrained_model_name = pretrained_model_name
        self.max_len = max_len
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.threshold = threshold
        self.random_state = random_state
        self.device = _get_device(device)
        self.verbose = verbose

        self.classes_ = np.array([0, 1])
        self.tokenizer_ = None
        self.model_ = None

    def _prepare_loader(self, texts, labels=None, shuffle=False):
        texts = [str(t) for t in texts]

        encodings = self.tokenizer_(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        if labels is not None:
            labels = torch.tensor(labels, dtype=torch.long)
            dataset = TensorDataset(input_ids, attention_mask, labels)
        else:
            dataset = TensorDataset(input_ids, attention_mask)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def fit(self, X, y):
        _set_seed(self.random_state)

        X = np.asarray(X, dtype=object)
        y = np.asarray(y, dtype=int)

        self.tokenizer_ = AutoTokenizer.from_pretrained(self.pretrained_model_name)
        self.model_ = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_model_name,
            num_labels=2,
        ).to(self.device)

        train_loader = self._prepare_loader(X, y, shuffle=True)

        class_counts = np.bincount(y, minlength=2).astype(np.float32)
        class_counts[class_counts == 0] = 1.0
        class_weights = class_counts.sum() / (2.0 * class_counts)
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=self.device)

        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr)

        self.model_.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch in train_loader:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]

                optimizer.zero_grad()
                outputs = self.model_(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * input_ids.size(0)

            if self.verbose:
                avg_loss = epoch_loss / len(train_loader.dataset)
                print(f"[DistilBERT] epoch {epoch + 1}/{self.epochs} - loss={avg_loss:.4f}")

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=object)
        loader = self._prepare_loader(X, labels=None, shuffle=False)

        self.model_.eval()
        probs = []
        with torch.no_grad():
            for batch in loader:
                input_ids, attention_mask = [b.to(self.device) for b in batch]
                outputs = self.model_(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                batch_probs = torch.softmax(logits, dim=1).cpu().numpy()
                probs.append(batch_probs)

        return np.vstack(probs)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
