from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Question:
    id: str
    type: str           # "mcq" | "true_false" | "fill_blank" | "code_snippet"
    topic: str
    difficulty: str
    question: str
    explanation: str
    tags: list[str] = field(default_factory=list)
    # MCQ / code_snippet fields
    options: list[str] | None = None
    correct_index: int | None = None
    # True/False
    correct_answer: bool | None = None
    # Fill in blank
    blank_marker: str | None = None
    correct_answers: list[str] | None = None
    case_sensitive: bool = False
    # Code snippet
    language: str | None = None
    preamble: str | None = None
    code_block: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Question":
        return cls(
            id=d["id"],
            type=d["type"],
            topic=d["topic"],
            difficulty=d["difficulty"],
            question=d["question"],
            explanation=d["explanation"],
            tags=d.get("tags", []),
            options=d.get("options"),
            correct_index=d.get("correct_index"),
            correct_answer=d.get("correct_answer"),
            blank_marker=d.get("blank_marker"),
            correct_answers=d.get("correct_answers"),
            case_sensitive=d.get("case_sensitive", False),
            language=d.get("language"),
            preamble=d.get("preamble"),
            code_block=d.get("code_block"),
        )


@dataclass
class SessionResult:
    session_id: str
    timestamp: str
    topic: str
    difficulty: str
    questions_attempted: int
    questions_correct: int
    xp_earned: int
    duration_seconds: int

    @property
    def accuracy(self) -> float:
        if self.questions_attempted == 0:
            return 0.0
        return self.questions_correct / self.questions_attempted
