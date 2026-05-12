from enum import Enum
from typing import Optional

from pydantic import BaseModel


class SensitivityCategory(str, Enum):
    RELIGION_PHILOSOPHY = "Religion and Philosophy"
    POLITICS_SOCIETY = "Politics and Society"
    RELATIONSHIPS_SENTIMENTS = "Relationships and Sentiments"
    HEALTH_MENTAL_WELLBEING = "Health and Mental Well-being"
    IDENTITY_DIVERSITY = "Identity and Diversity"
    SEXUAL = "Sexual"
    SECURITY = "Security"
    OTHER = "Other"


class DatasetPrompt(BaseModel):
    """
    Rappresenta un prompt generico del dataset SensY 2.0.

    Può essere:
    - sensitive = 1
    - sensitive = 0

    Nel dataset originale i prompt non-sensitive di solito non hanno categoria.
    Per questo motivo:
    - se sensitive = 0, categories può essere []
    - se sensitive = 1, categories dovrebbe contenere almeno una categoria
      oppure OTHER nei casi anomali.

    Questa classe serve per caricare tutto il dataset in modo uniforme.
    """

    prompt_id: str
    text: str
    sensitive: int

    # Lista obbligatoria.
    # Il loader deve sempre passarla:
    # - [] per prompt non-sensitive;
    # - una o più categorie per prompt sensitive;
    # - [OTHER] per prompt sensitive senza categoria valida.
    categories: list[SensitivityCategory]

    raw_category: Optional[str] = None
    source: str = "SensY2.0"


class SensitivePrompt(BaseModel):
    """
    Rappresenta un prompt sensibile proveniente da SensY 2.0.

    Questa è l'unità di input principale della RQ2.

    In SensY 2.0:
    - solo i prompt sensibili hanno il campo category;
    - alcuni prompt possono avere più categorie nello stesso campo,
      ad esempio: "politics and society/security".

    Per questo motivo usiamo categories: list[SensitivityCategory]
    invece di una singola category.
    """

    prompt_id: str
    text: str
    sensitive: int = 1

    # Lista obbligatoria.
    # Il loader la deve sempre passare.
    categories: list[SensitivityCategory]

    raw_category: Optional[str] = None
    subcategory: Optional[str] = None
    style_type: Optional[str] = None
    source: str = "SensY2.0"