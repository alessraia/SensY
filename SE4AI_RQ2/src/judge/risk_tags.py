from enum import Enum


class RiskTag(str, Enum):
    """
    Risk tag operativi usati per selezionare le rubriche modulari.

    Questi tag non coincidono necessariamente con le categorie SensY.
    Sono più specifici e servono a capire quale tipo di rischio valutativo
    deve essere considerato dal judge.
    """

    PRIVACY_SENSITIVE_DATA = "privacy_sensitive_data"

    HEALTH_ADVICE = "health_advice"
    MENTAL_HEALTH = "mental_health"
    BODY_DIET_FITNESS = "body_diet_fitness"

    FINANCIAL_RISK = "financial_risk"

    IDENTITY_STEREOTYPE = "identity_stereotype"

    RELATIONSHIP_MANIPULATION = "relationship_manipulation"
    DECEPTION = "deception"

    UNSAFE_SECURITY = "unsafe_security"

    SEXUAL_SENSITIVE = "sexual_sensitive"

    POLITICAL_SENSITIVE = "political_sensitive"
    GEOPOLITICAL_SENSITIVE = "geopolitical_sensitive"
    RELIGIOUS_SENSITIVE = "religious_sensitive"

    HIGH_UNCERTAINTY = "high_uncertainty"

    GENERAL_SENSITIVE = "general_sensitive"