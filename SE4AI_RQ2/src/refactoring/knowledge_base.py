import json
from pathlib import Path

from refactoring_models import (
    PromptRefactoringPattern,
    PromptSmellDefinition,
    RefactoringKnowledgeBase,
)


DEFAULT_SMELLS_PATH = Path("config/prompt_smells_sensitive_prompts.json")
DEFAULT_PATTERNS_PATH = Path("config/prompt_refactoring_patterns.json")


class KnowledgeBaseValidationError(Exception):
    """
    Raised when the refactoring knowledge base is malformed or inconsistent.
    """


def _load_json_array(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise KnowledgeBaseValidationError(
            f"Expected a JSON array in {path}, found {type(data).__name__}."
        )

    return data


def load_refactoring_knowledge_base(
    smells_path: str | Path = DEFAULT_SMELLS_PATH,
    patterns_path: str | Path = DEFAULT_PATTERNS_PATH,
) -> RefactoringKnowledgeBase:
    """
    Loads and validates the prompt smells and refactoring patterns catalogs.

    This function does not call any LLM and does not modify experimental data.
    It only verifies that the two knowledge base files are structurally valid
    and mutually consistent.
    """

    smells_path = Path(smells_path)
    patterns_path = Path(patterns_path)

    smell_records = _load_json_array(smells_path)
    pattern_records = _load_json_array(patterns_path)

    smells = [
        PromptSmellDefinition(**record)
        for record in smell_records
    ]

    patterns = [
        PromptRefactoringPattern(**record)
        for record in pattern_records
    ]

    knowledge_base = RefactoringKnowledgeBase(
        smells=smells,
        patterns=patterns,
    )

    validate_knowledge_base(knowledge_base)

    return knowledge_base


def validate_knowledge_base(
    knowledge_base: RefactoringKnowledgeBase,
) -> None:
    """
    Performs consistency checks on the knowledge base.

    Checks:
    - smell identifiers must be unique;
    - pattern identifiers must be unique;
    - every related smell referenced by a pattern must exist;
    - the fallback pattern should exist.
    """

    smell_ids = [smell.smell_id for smell in knowledge_base.smells]
    pattern_ids = [pattern.pattern_id for pattern in knowledge_base.patterns]

    duplicated_smells = _find_duplicates(smell_ids)
    duplicated_patterns = _find_duplicates(pattern_ids)

    if duplicated_smells:
        raise KnowledgeBaseValidationError(
            f"Duplicated smell_id values: {sorted(duplicated_smells)}"
        )

    if duplicated_patterns:
        raise KnowledgeBaseValidationError(
            f"Duplicated pattern_id values: {sorted(duplicated_patterns)}"
        )

    known_smell_ids = set(smell_ids)

    for pattern in knowledge_base.patterns:
        for related_smell in pattern.related_smells:
            if related_smell not in known_smell_ids:
                raise KnowledgeBaseValidationError(
                    "Pattern "
                    f"'{pattern.pattern_id}' references unknown smell_id "
                    f"'{related_smell}'."
                )

    if knowledge_base.get_fallback_pattern() is None:
        raise KnowledgeBaseValidationError(
            "Missing fallback pattern: generic_responsible_refactoring"
        )


def _find_duplicates(values: list[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()

    for value in values:
        if value in seen:
            duplicates.add(value)
        else:
            seen.add(value)

    return duplicates


def knowledge_base_to_prompt_context(
    knowledge_base: RefactoringKnowledgeBase,
) -> str:
    """
    Converts the knowledge base into a compact textual context to be included
    in the future refactoring-model prompt.

    This is not used yet in the current step, but it will be useful when
    implementing the script that calls Qwen3-14B.
    """

    lines: list[str] = []

    lines.append("KNOWN PROMPT SMELLS")
    for smell in knowledge_base.smells:
        lines.append(f"- {smell.smell_id}: {smell.name}")
        lines.append(f"  Description: {smell.description}")
        lines.append(f"  Observable cues: {'; '.join(smell.observable_cues)}")

    lines.append("")
    lines.append("KNOWN PROMPT REFACTORING PATTERNS")
    for pattern in knowledge_base.patterns:
        related = ", ".join(pattern.related_smells) if pattern.related_smells else "none"
        lines.append(f"- {pattern.pattern_id}: {pattern.name}")
        lines.append(f"  Related smells: {related}")
        lines.append(f"  Transformation: {pattern.transformation}")
        lines.append(f"  Intent preservation rule: {pattern.intent_preservation_rule}")

    return "\n".join(lines)