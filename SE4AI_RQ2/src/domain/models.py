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


class AdequacyLabel(str, Enum):
    ADEQUATE = "adequate"
    INADEQUATE = "inadequate"


class DatasetPrompt(BaseModel):
    """
    Rappresenta un prompt generico del dataset SensY 2.0.
    Può essere sensitive oppure non-sensitive.
    """

    prompt_id: str
    text: str
    sensitive: int

    categories: list[SensitivityCategory]

    raw_category: Optional[str] = None
    category_present: bool = False

    source: str = "SensY2.0"


class SensitivePrompt(BaseModel):
    """
    Rappresenta un prompt sensibile proveniente da SensY 2.0.
    Questa è l'unità di input principale della RQ2.
    """

    prompt_id: str
    text: str
    sensitive: int = 1

    categories: list[SensitivityCategory]

    raw_category: Optional[str] = None
    category_present: bool = False

    subcategory: Optional[str] = None
    style_type: Optional[str] = None
    source: str = "SensY2.0"


class TargetResponse(BaseModel):
    """
    Rappresenta una risposta prodotta da un modello target.

    Nello Sprint 2B è stata usata per salvare le risposte baseline.
    Più avanti sarà riutilizzata anche per le risposte post-refactoring.
    """

    response_id: str
    prompt_id: str

    target_model: str
    repetition: int

    prompt_text: str
    response_text: str

    source: str = "baseline"


class JudgeResponseEvaluationPayload(BaseModel):
    """
    Payload grezzo restituito dal judge.

    Non contiene gli ID sperimentali, perché questi vengono aggiunti
    dal codice dopo il parsing dell'output JSON del judge.
    """

    adequacy_score: int
    safety_score: int
    helpfulness_score: int
    sensitivity_awareness_score: int

    over_refusal: bool
    under_refusal: bool

    label: AdequacyLabel
    rationale: str
    failure_modes: list[str]


class ResponseEvaluation(BaseModel):
    """
    Valutazione strutturata di una risposta baseline.

    Questa classe rappresenta il giudizio completo del LLM-as-a-Judge
    su una risposta prodotta dal target model.
    """

    evaluation_id: str
    response_id: str
    prompt_id: str

    target_model: str
    judge_model: str
    repetition: int

    adequacy_score: int
    safety_score: int
    helpfulness_score: int
    sensitivity_awareness_score: int

    over_refusal: bool
    under_refusal: bool

    label: AdequacyLabel
    rationale: str
    failure_modes: list[str]

    source: str = "baseline_evaluation"


class PromptSeedDecision(BaseModel):
    """
    Decisione finale a livello di prompt.

    La valutazione avviene a livello di risposta, ma SensY-Refactor seed
    viene costruito a livello di prompt.

    Regola iniziale:
    un prompt entra nel seed se almeno due delle sue tre risposte baseline
    sono valutate come inadeguate.
    """

    prompt_id: str
    text: str

    categories: list[SensitivityCategory]
    raw_category: Optional[str] = None

    target_model: str
    total_responses: int
    inadequate_responses: int
    adequate_responses: int

    enters_refactor_seed: bool
    decision_rule: str

class HumanLabeledResponse(BaseModel):
    """
    Rappresenta una singola risposta etichettata manualmente.

    Questa classe viene usata nello Sprint 2C.0 per costruire
    l'oracolo umano usato nella calibrazione del LLM-as-a-Judge.
    """

    oracle_id: str

    question: str
    category: Optional[str] = None

    response_text: str

    manual_adequate: int
    manual_label: AdequacyLabel

    source_model: str
    response_index: int

    oracle_source_file: str


class OracleJudgeEvaluation(BaseModel):
    """
    Valutazione prodotta dal LLM-as-a-Judge su una risposta
    già etichettata manualmente.

    Serve per confrontare il giudizio automatico con l'oracolo umano.
    """

    oracle_evaluation_id: str
    oracle_id: str

    question: str
    category: Optional[str] = None
    response_text: str

    source_model: str
    response_index: int
    oracle_source_file: str

    manual_adequate: int
    manual_label: AdequacyLabel

    judge_model: str

    adequacy_score: int
    safety_score: int
    helpfulness_score: int
    sensitivity_awareness_score: int

    over_refusal: bool
    under_refusal: bool

    judge_label: AdequacyLabel
    rationale: str
    failure_modes: list[str]

    source: str = "oracle_judge_evaluation"