from .risk_tags import RiskTag
from .rubrics import (
    BODY_DIET_FITNESS_ADDENDUM,
    CORE_RUBRIC,
    DECEPTION_ADDENDUM,
    FINANCIAL_ADDENDUM,
    GENERAL_SENSITIVE_ADDENDUM,
    GEOPOLITICAL_ADDENDUM,
    HEALTH_ADDENDUM,
    HIGH_UNCERTAINTY_ADDENDUM,
    IDENTITY_BIAS_ADDENDUM,
    MENTAL_HEALTH_ADDENDUM,
    POLITICAL_ADDENDUM,
    PRIVACY_ADDENDUM,
    RELATIONSHIP_ADDENDUM,
    RELIGIOUS_ADDENDUM,
    SECURITY_ADDENDUM,
    SEXUAL_ADDENDUM,
)


class RubricBuilder:
    """
    Costruisce la rubrica finale del judge.

    La rubrica è composta da:
    - CORE_RUBRIC sempre presente;
    - addenda specifici selezionati in base ai risk tag.
    """

    def __init__(self, max_addenda: int = 4):
        self.max_addenda = max_addenda

        self.addendum_by_tag = {
            RiskTag.PRIVACY_SENSITIVE_DATA: PRIVACY_ADDENDUM,
            RiskTag.HEALTH_ADVICE: HEALTH_ADDENDUM,
            RiskTag.MENTAL_HEALTH: MENTAL_HEALTH_ADDENDUM,
            RiskTag.BODY_DIET_FITNESS: BODY_DIET_FITNESS_ADDENDUM,
            RiskTag.FINANCIAL_RISK: FINANCIAL_ADDENDUM,
            RiskTag.IDENTITY_STEREOTYPE: IDENTITY_BIAS_ADDENDUM,
            RiskTag.RELATIONSHIP_MANIPULATION: RELATIONSHIP_ADDENDUM,
            RiskTag.UNSAFE_SECURITY: SECURITY_ADDENDUM,
            RiskTag.SEXUAL_SENSITIVE: SEXUAL_ADDENDUM,
            RiskTag.POLITICAL_SENSITIVE: POLITICAL_ADDENDUM,
            RiskTag.GEOPOLITICAL_SENSITIVE: GEOPOLITICAL_ADDENDUM,
            RiskTag.RELIGIOUS_SENSITIVE: RELIGIOUS_ADDENDUM,
            RiskTag.DECEPTION: DECEPTION_ADDENDUM,
            RiskTag.HIGH_UNCERTAINTY: HIGH_UNCERTAINTY_ADDENDUM,
            RiskTag.GENERAL_SENSITIVE: GENERAL_SENSITIVE_ADDENDUM,
        }

        self.priority_order = [
            RiskTag.PRIVACY_SENSITIVE_DATA,
            RiskTag.UNSAFE_SECURITY,
            RiskTag.HEALTH_ADVICE,
            RiskTag.MENTAL_HEALTH,
            RiskTag.BODY_DIET_FITNESS,
            RiskTag.IDENTITY_STEREOTYPE,
            RiskTag.RELATIONSHIP_MANIPULATION,
            RiskTag.SEXUAL_SENSITIVE,
            RiskTag.FINANCIAL_RISK,
            RiskTag.DECEPTION,
            RiskTag.GEOPOLITICAL_SENSITIVE,
            RiskTag.POLITICAL_SENSITIVE,
            RiskTag.RELIGIOUS_SENSITIVE,
            RiskTag.HIGH_UNCERTAINTY,
            RiskTag.GENERAL_SENSITIVE,
        ]

    def select_tags(self, risk_tags: list[RiskTag]) -> list[RiskTag]:
        """
        Seleziona un sottoinsieme ordinato di risk tag.

        Evita rubriche troppo lunghe quando il router assegna molti tag.
        GENERAL_SENSITIVE viene usato solo se è l'unico tag disponibile.
        """

        unique_tags = list(dict.fromkeys(risk_tags))

        if not unique_tags:
            return [RiskTag.GENERAL_SENSITIVE]

        non_general_tags = [
            tag for tag in unique_tags if tag != RiskTag.GENERAL_SENSITIVE
        ]

        if not non_general_tags:
            return [RiskTag.GENERAL_SENSITIVE]

        ordered_tags = [
            tag for tag in self.priority_order if tag in non_general_tags
        ]

        return ordered_tags[: self.max_addenda]

    def build(self, risk_tags: list[RiskTag]) -> str:
        selected_tags = self.select_tags(risk_tags)

        sections = [
            CORE_RUBRIC,
            self._format_selected_tags(selected_tags),
        ]

        for tag in selected_tags:
            addendum = self.addendum_by_tag.get(tag)

            if addendum:
                sections.append(addendum)

        return "\n\n".join(sections).strip()

    def _format_selected_tags(self, selected_tags: list[RiskTag]) -> str:
        tag_values = [tag.value for tag in selected_tags]

        return (
            "Risk tags selected for this evaluation:\n"
            + "\n".join(f"- {tag}" for tag in tag_values)
        )