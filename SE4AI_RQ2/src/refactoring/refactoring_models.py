from typing import Optional

from pydantic import BaseModel, Field


class PromptSmellDefinition(BaseModel):
    """
    Definition of a prompt smell used in the smell-guided refactoring pipeline.

    A prompt smell is treated as a structured hypothesis about a prompt-level
    issue that may increase the likelihood of inadequate responses.
    It is not interpreted as a ground-truth causal explanation.
    """

    smell_id: str
    name: str
    description: str
    sensitive_prompt_rationale: str

    typical_categories: list[str] = Field(default_factory=list)
    observable_cues: list[str] = Field(default_factory=list)
    associated_failure_risks: list[str] = Field(default_factory=list)


class PromptRefactoringPattern(BaseModel):
    """
    Reusable prompt transformation strategy.

    A refactoring pattern specifies how to modify the prompt while preserving
    the original semantic intent and improving the expected adequacy of the
    target model response.
    """

    pattern_id: str
    name: str

    problem_addressed: str
    context_of_application: str

    related_smells: list[str] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)

    transformation: str
    intent_preservation_rule: str
    expected_effect: str

    before_example: Optional[str] = None
    after_example: Optional[str] = None


class RefactoringKnowledgeBase(BaseModel):
    """
    In-memory representation of the refactoring knowledge base.

    It contains:
    - a catalog of prompt smells specialized for sensitive prompts;
    - a catalog of refactoring patterns that can be selected by the
      refactoring model.
    """

    smells: list[PromptSmellDefinition]
    patterns: list[PromptRefactoringPattern]

    def get_smell_ids(self) -> set[str]:
        return {smell.smell_id for smell in self.smells}

    def get_pattern_ids(self) -> set[str]:
        return {pattern.pattern_id for pattern in self.patterns}

    def get_patterns_for_smell(self, smell_id: str) -> list[PromptRefactoringPattern]:
        return [
            pattern
            for pattern in self.patterns
            if smell_id in pattern.related_smells
        ]

    def get_fallback_pattern(self) -> Optional[PromptRefactoringPattern]:
        for pattern in self.patterns:
            if pattern.pattern_id == "generic_responsible_refactoring":
                return pattern

        return None


class RefactoringKnownSmell(BaseModel):
    """
    Smell proposed by the refactoring model.

    This is not treated as ground truth. It is a smell-based diagnosis proposed
    by the refactoring model using the knowledge base.
    """

    smell_id: str
    smell_name: str
    confidence: str
    rationale: str


class RefactoringSelectedPattern(BaseModel):
    """
    Refactoring pattern selected by the refactoring model.
    """

    pattern_id: str
    pattern_name: str
    rationale: str


class UncoveredIssue(BaseModel):
    """
    Issue proposed when no known smell fully covers the prompt-level problem.

    This is not automatically considered a new prompt smell. It is only an
    uncovered issue that may later become a candidate emergent smell if it
    recurs and is manually validated.
    """

    candidate_name: str
    description: str
    why_known_smells_do_not_fit: str
    suggested_refactoring_strategy: str


class RefactoringAttemptPayload(BaseModel):
    """
    Raw structured payload expected from the refactoring model.

    The script enriches this payload with identifiers, iteration metadata,
    model name, and traceability fields.
    """

    known_smells_detected: list[RefactoringKnownSmell] = Field(default_factory=list)
    selected_patterns: list[RefactoringSelectedPattern] = Field(default_factory=list)

    uncovered_issue: Optional[UncoveredIssue] = None

    refactored_prompt: str
    semantic_preservation_rationale: str
    expected_effect: str


class RefactoringAttempt(BaseModel):
    """
    Final record saved by the refactoring pipeline.

    One record corresponds to one refactoring attempt for one prompt.

    Traceability fields are optional to preserve compatibility with older
    JSONL files already generated before the introduction of the episode model.
    """

    refactoring_attempt_id: str

    prompt_id: str
    iteration: int

    original_prompt: str
    category: str

    known_smells_detected: list[RefactoringKnownSmell] = Field(default_factory=list)
    selected_patterns: list[RefactoringSelectedPattern] = Field(default_factory=list)

    uncovered_issue: Optional[UncoveredIssue] = None

    refactored_prompt: str
    semantic_preservation_rationale: str
    expected_effect: str

    refactoring_model: str
    refactoring_method: str = "smell_guided_refactoring_v1"

    source: str = "refactoring_attempt"

    # Traceability fields for iterative refactoring.
    episode_id: Optional[str] = None
    parent_refactoring_attempt_id: Optional[str] = None
    input_prompt: Optional[str] = None
    input_prompt_source: Optional[str] = None
    iteration_reason: Optional[str] = None

    @property
    def effective_episode_id(self) -> str:
        """
        Returns the explicit episode_id when available, otherwise derives it
        from the prompt_id for backward compatibility.
        """
        return self.episode_id or f"episode_{self.prompt_id}"

    @property
    def effective_input_prompt(self) -> str:
        """
        Returns the prompt that was used as input for this refactoring attempt.
        For older records, this falls back to the original prompt.
        """
        return self.input_prompt or self.original_prompt


class SemanticPreservationPayload(BaseModel):
    """
    Raw structured payload expected from the semantic preservation evaluator.

    The evaluator compares the original prompt and the refactored prompt and
    determines whether the refactoring preserved the original semantic intent.
    """

    topic_preservation_score: int = Field(ge=1, le=5)
    goal_preservation_score: int = Field(ge=1, le=5)
    task_type_preservation_score: int = Field(ge=1, le=5)
    specificity_preservation_score: int = Field(ge=1, le=5)
    safety_framing_score: int = Field(ge=1, le=5)

    semantic_preservation_score: int = Field(ge=1, le=5)

    semantic_shift_detected: bool
    semantic_shift_description: Optional[str] = None

    decision: str
    rationale: str


class SemanticPreservationEvaluation(BaseModel):
    """
    Final semantic preservation evaluation saved by the pipeline.

    One record corresponds to one refactoring attempt.
    """

    semantic_evaluation_id: str

    refactoring_attempt_id: str
    prompt_id: str
    iteration: int

    original_prompt: str
    refactored_prompt: str
    category: str

    topic_preservation_score: int = Field(ge=1, le=5)
    goal_preservation_score: int = Field(ge=1, le=5)
    task_type_preservation_score: int = Field(ge=1, le=5)
    specificity_preservation_score: int = Field(ge=1, le=5)
    safety_framing_score: int = Field(ge=1, le=5)

    semantic_preservation_score: int = Field(ge=1, le=5)

    semantic_shift_detected: bool
    semantic_shift_description: Optional[str] = None

    decision: str
    rationale: str

    evaluator_model: str
    evaluation_method: str = "semantic_preservation_v1"

    source: str = "semantic_preservation_evaluation"

    # Traceability fields.
    episode_id: Optional[str] = None
    parent_refactoring_attempt_id: Optional[str] = None
    input_prompt_source: Optional[str] = None
    iteration_reason: Optional[str] = None