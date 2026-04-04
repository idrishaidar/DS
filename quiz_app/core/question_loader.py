from __future__ import annotations
import json
from pathlib import Path

import streamlit as st

from core.constants import TOPICS, DIFFICULTIES

DATA_DIR = Path(__file__).parent.parent / "data"

REQUIRED_FIELDS = {"id", "type", "difficulty", "topic", "explanation"}
# code_snippet uses 'preamble' instead of 'question'
TYPE_REQUIRED: dict[str, set[str]] = {
    "mcq":          {"question", "options", "correct_index"},
    "true_false":   {"question", "correct_answer"},
    "fill_blank":   {"question", "blank_marker", "correct_answers"},
    "code_snippet": {"language", "preamble", "code_block", "options", "correct_index"},
}


def _validate_question(q: dict) -> None:
    missing = REQUIRED_FIELDS - q.keys()
    if missing:
        raise ValueError(f"Question {q.get('id', '?')} missing fields: {missing}")
    qtype = q["type"]
    if qtype not in TYPE_REQUIRED:
        raise ValueError(f"Question {q['id']}: unknown type '{qtype}'")
    type_missing = TYPE_REQUIRED[qtype] - q.keys()
    if type_missing:
        raise ValueError(f"Question {q['id']} ({qtype}) missing: {type_missing}")


def load_questions(topic: str, difficulty: str) -> list[dict]:
    path = DATA_DIR / topic / f"{difficulty}.json"
    if not path.exists():
        return []
    with open(path) as f:
        bank = json.load(f)
    for q in bank.get("questions", []):
        _validate_question(q)
    return bank["questions"]


@st.cache_data(show_spinner=False)
def load_all_questions() -> dict[str, dict[str, list[dict]]]:
    """Returns {topic: {difficulty: [questions]}}. Cached per Streamlit session."""
    result: dict[str, dict[str, list[dict]]] = {}
    for topic in TOPICS:
        result[topic] = {}
        for diff in DIFFICULTIES:
            result[topic][diff] = load_questions(topic, diff)
    return result


def question_counts(all_questions: dict[str, dict[str, list[dict]]]) -> dict[str, dict[str, int]]:
    return {
        topic: {diff: len(qs) for diff, qs in diffs.items()}
        for topic, diffs in all_questions.items()
    }
