from pathlib import Path
from typing import Optional

import pandas as pd

from ..data.category_utils import normalize_categories
from ..domain.models import DatasetPrompt, SensitivePrompt, SensitivityCategory


class SensYDatasetLoader:
    """
    Carica SensY 2.0 e normalizza le colonne principali.

    Formato atteso del dataset:

    [
        {
            "question_en": "What is the difference between OpenCL and CUDA?",
            "sensitive?": 0
        },
        {
            "question_en": "Why did my parent not invite me to their wedding?",
            "sensitive?": 1,
            "category": "relationships and sentiments"
        },
        {
            "question_en": "...",
            "sensitive?": 1,
            "category": "politics and society/security"
        }
    ]

    Note:
    - i prompt non-sensitive possono non avere category;
    - i prompt sensitive dovrebbero avere category;
    - alcuni prompt sensitive possono avere più categorie.
    """

    TEXT_COLUMN_CANDIDATES = [
        "question_en",
        "prompt",
        "text",
        "question",
        "input",
    ]

    LABEL_COLUMN_CANDIDATES = [
        "sensitive?",
        "sensitive",
        "label",
        "is_sensitive",
    ]

    CATEGORY_COLUMN_CANDIDATES = [
        "category",
        "macro_category",
        "sensitivity_category",
    ]

    ID_COLUMN_CANDIDATES = [
        "id",
        "prompt_id",
        "question_id",
    ]

    def load_dataframe(self, path: str | Path) -> pd.DataFrame:
        """
        Carica il dataset da CSV, JSON o JSONL.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        suffix = path.suffix.lower()

        if suffix == ".csv":
            return pd.read_csv(path)

        if suffix == ".json":
            return pd.read_json(path)

        if suffix == ".jsonl":
            return pd.read_json(path, lines=True)

        raise ValueError(f"Unsupported dataset format: {path.suffix}")

    def find_column(
        self,
        df: pd.DataFrame,
        candidates: list[str],
        required: bool = True,
    ) -> Optional[str]:
        """
        Cerca una colonna nel DataFrame tra vari nomi possibili.
        """

        columns_lower = {col.lower(): col for col in df.columns}

        for candidate in candidates:
            if candidate.lower() in columns_lower:
                return columns_lower[candidate.lower()]

        if required:
            raise ValueError(
                f"None of the candidate columns were found: {candidates}. "
                f"Available columns: {list(df.columns)}"
            )

        return None

    def _is_sensitive_value(self, value) -> bool:
        """
        Gestisce label scritte come:
        - 1
        - "1"
        - true
        - "true"
        - "sensitive"
        - "yes"
        """

        normalized = str(value).strip().lower()
        return normalized in {"1", "true", "sensitive", "yes"}

    def load_all_prompts(self, path: str | Path) -> list[DatasetPrompt]:
        """
        Carica tutti i prompt, sensitive e non-sensitive.

        Utile per controlli globali sul dataset.
        Per RQ2 useremo poi load_sensitive_prompts().
        """

        df = self.load_dataframe(path)

        text_col = self.find_column(df, self.TEXT_COLUMN_CANDIDATES)
        label_col = self.find_column(df, self.LABEL_COLUMN_CANDIDATES)

        category_col = self.find_column(
            df,
            self.CATEGORY_COLUMN_CANDIDATES,
            required=False,
        )

        id_col = self.find_column(
            df,
            self.ID_COLUMN_CANDIDATES,
            required=False,
        )

        prompts: list[DatasetPrompt] = []

        for index, row in df.iterrows():
            prompt_id = str(row[id_col]) if id_col is not None else f"sensy2_{index}"
            text = str(row[text_col]).strip()

            if not text:
                continue

            sensitive = 1 if self._is_sensitive_value(row[label_col]) else 0

            raw_category = None
            categories: list[SensitivityCategory] = []

            if category_col is not None and category_col in row and pd.notna(row[category_col]):
                raw_category = str(row[category_col]).strip()
                categories = normalize_categories(raw_category)

            # Caso importante:
            # se il prompt è sensitive ma non ha categoria valida,
            # non lasciamo categories vuota, ma assegniamo OTHER.
            if sensitive == 1 and not categories:
                categories = [SensitivityCategory.OTHER]

            # Caso non-sensitive:
            # se non ha categoria, categories resta [].
            # Questo è coerente con SensY 2.0.
            prompt = DatasetPrompt(
                prompt_id=prompt_id,
                text=text,
                sensitive=sensitive,
                categories=categories,
                raw_category=raw_category,
                source="SensY2.0",
            )

            prompts.append(prompt)

        return prompts

    def load_sensitive_prompts(self, path: str | Path) -> list[SensitivePrompt]:
        """
        Carica solo i prompt sensitive.

        I prompt non-sensitive vengono ignorati perché la RQ2 lavora
        sulla mitigazione/refactoring dei prompt sensibili.
        """

        all_prompts = self.load_all_prompts(path)

        sensitive_prompts: list[SensitivePrompt] = []

        for prompt in all_prompts:
            if prompt.sensitive != 1:
                continue

            sensitive_prompt = SensitivePrompt(
                prompt_id=prompt.prompt_id,
                text=prompt.text,
                sensitive=1,
                categories=prompt.categories,
                raw_category=prompt.raw_category,
                source=prompt.source,
            )

            sensitive_prompts.append(sensitive_prompt)

        return sensitive_prompts