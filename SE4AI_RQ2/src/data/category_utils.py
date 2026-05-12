import re
from ..domain.models import SensitivityCategory


CATEGORY_ALIASES = {
    # Religion and Philosophy
    "religion and philosophy": SensitivityCategory.RELIGION_PHILOSOPHY,
    "religion philosophy": SensitivityCategory.RELIGION_PHILOSOPHY,
    "religion_philosophy": SensitivityCategory.RELIGION_PHILOSOPHY,
    "religion": SensitivityCategory.RELIGION_PHILOSOPHY,
    "philosophy": SensitivityCategory.RELIGION_PHILOSOPHY,

    # Politics and Society
    "politics and society": SensitivityCategory.POLITICS_SOCIETY,
    "politics society": SensitivityCategory.POLITICS_SOCIETY,
    "politics_society": SensitivityCategory.POLITICS_SOCIETY,
    "politics": SensitivityCategory.POLITICS_SOCIETY,
    "society": SensitivityCategory.POLITICS_SOCIETY,

    # Relationships and Sentiments
    "relationships and sentiments": SensitivityCategory.RELATIONSHIPS_SENTIMENTS,
    "relationship and sentiments": SensitivityCategory.RELATIONSHIPS_SENTIMENTS,
    "relationships sentiments": SensitivityCategory.RELATIONSHIPS_SENTIMENTS,
    "relationships_sentiments": SensitivityCategory.RELATIONSHIPS_SENTIMENTS,
    "relationships": SensitivityCategory.RELATIONSHIPS_SENTIMENTS,
    "relationship": SensitivityCategory.RELATIONSHIPS_SENTIMENTS,
    "sentiments": SensitivityCategory.RELATIONSHIPS_SENTIMENTS,

    # Health and Mental Well-being
    "health and mental well-being": SensitivityCategory.HEALTH_MENTAL_WELLBEING,
    "health and mental wellbeing": SensitivityCategory.HEALTH_MENTAL_WELLBEING,
    "health mental well-being": SensitivityCategory.HEALTH_MENTAL_WELLBEING,
    "health mental wellbeing": SensitivityCategory.HEALTH_MENTAL_WELLBEING,
    "health_mental_wellbeing": SensitivityCategory.HEALTH_MENTAL_WELLBEING,
    "health": SensitivityCategory.HEALTH_MENTAL_WELLBEING,
    "mental well-being": SensitivityCategory.HEALTH_MENTAL_WELLBEING,
    "mental wellbeing": SensitivityCategory.HEALTH_MENTAL_WELLBEING,

    # Identity and Diversity
    "identity and diversity": SensitivityCategory.IDENTITY_DIVERSITY,
    "identity diversity": SensitivityCategory.IDENTITY_DIVERSITY,
    "identity_diversity": SensitivityCategory.IDENTITY_DIVERSITY,
    "identity": SensitivityCategory.IDENTITY_DIVERSITY,
    "diversity": SensitivityCategory.IDENTITY_DIVERSITY,

    # Sexual
    "sexual": SensitivityCategory.SEXUAL,
    "sex": SensitivityCategory.SEXUAL,

    # Security
    "security": SensitivityCategory.SECURITY,
}


def normalize_category(value: str | None) -> SensitivityCategory:
    """
    Normalizza una singola categoria testuale.

    Esempio:
    "relationships and sentiments"
        -> SensitivityCategory.RELATIONSHIPS_SENTIMENTS
    """

    if value is None:
        return SensitivityCategory.OTHER

    normalized = str(value).strip().lower()

    if not normalized:
        return SensitivityCategory.OTHER

    if normalized in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[normalized]

    for category in SensitivityCategory:
        if normalized == category.value.lower():
            return category

    return SensitivityCategory.OTHER


def normalize_categories(value: str | None) -> list[SensitivityCategory]:
    """
    Normalizza il campo category di SensY 2.0.

    Gestisce categorie singole:

        "relationships and sentiments"

    e categorie multiple:

        "politics and society/security"

    Restituisce sempre una lista.
    """

    if value is None:
        return []

    raw = str(value).strip()

    if not raw:
        return []

    # Separiamo su "/" perché il caso noto è:
    # "politics and society/security"
    #
    # Aggiungiamo anche ; e | per robustezza futura.
    parts = re.split(r"\s*/\s*|\s*;\s*|\s*\|\s*", raw)

    categories: list[SensitivityCategory] = []

    for part in parts:
        category = normalize_category(part)

        if category != SensitivityCategory.OTHER and category not in categories:
            categories.append(category)

    if not categories:
        return [SensitivityCategory.OTHER]

    return categories