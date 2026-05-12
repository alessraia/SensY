import random
from collections import defaultdict

from ..domain.models import SensitivePrompt, SensitivityCategory


class SensitivePromptSampler:
    """
    Crea un campione bilanciato per categoria primaria.

    Se un prompt appartiene a più categorie, per il campionamento viene usata
    la prima categoria normalizzata, ma l'informazione completa rimane salvata
    nel campo categories.

    Dopo aver selezionato n prompt per categoria, il campione finale viene
    mescolato per evitare che il file sia ordinato per blocchi di categoria.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def get_primary_category(self, prompt: SensitivePrompt) -> SensitivityCategory:
        if prompt.categories:
            return prompt.categories[0]

        return SensitivityCategory.OTHER

    def sample_balanced(
        self,
        prompts: list[SensitivePrompt],
        n_per_category: int,
        include_other: bool = False,
        shuffle_final: bool = True,
    ) -> list[SensitivePrompt]:
        rng = random.Random(self.random_state)

        grouped: dict[SensitivityCategory, list[SensitivePrompt]] = defaultdict(list)

        for prompt in prompts:
            primary_category = self.get_primary_category(prompt)

            if not include_other and primary_category == SensitivityCategory.OTHER:
                continue

            grouped[primary_category].append(prompt)

        sampled: list[SensitivePrompt] = []

        for category in SensitivityCategory:
            if category == SensitivityCategory.OTHER and not include_other:
                continue

            category_prompts = grouped.get(category, [])

            if not category_prompts:
                print(f"[WARNING] No prompts found for category: {category.value}")
                continue

            rng.shuffle(category_prompts)

            if len(category_prompts) < n_per_category:
                print(
                    f"[WARNING] Category '{category.value}' has only "
                    f"{len(category_prompts)} prompts; requested {n_per_category}."
                )
                sampled.extend(category_prompts)
            else:
                sampled.extend(category_prompts[:n_per_category])

        if shuffle_final:
            rng.shuffle(sampled)

        return sampled