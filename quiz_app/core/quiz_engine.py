from __future__ import annotations
import math
import random
from datetime import date

from core.constants import DIFFICULTIES, XP_TABLE, SPEED_BONUS_XP, SPEED_BONUS_SECONDS, streak_multiplier


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_questions(
    all_questions: dict,
    topic: str,
    difficulty: str,
    count: int,
    question_history: dict,
) -> list[dict]:
    """
    Returns a shuffled list of `count` questions.
    Deprioritises recently seen questions using (seen_count, last_seen_date).
    For difficulty="mixed", draws from all difficulties weighted 40/40/20.
    """
    if difficulty == "mixed":
        pool = _build_mixed_pool(all_questions, topic, count)
    else:
        pool = list(all_questions.get(topic, {}).get(difficulty, []))

    if not pool:
        return []

    def sort_key(q: dict) -> tuple:
        hist = question_history.get(q["id"], {})
        seen = hist.get("seen", 0)
        last = hist.get("last_seen") or "0000-00-00"
        return (seen, last)

    pool.sort(key=sort_key)
    # Take the least-seen first, then shuffle within equal-priority groups
    # to avoid always seeing the same first question.
    selected = pool[:max(count * 2, count + 10)]
    random.shuffle(selected)
    return selected[:count]


def _build_mixed_pool(all_questions: dict, topic: str, count: int) -> list[dict]:
    beg = list(all_questions.get(topic, {}).get("beginner", []))
    mid = list(all_questions.get(topic, {}).get("intermediate", []))
    adv = list(all_questions.get(topic, {}).get("advanced", []))

    # Target distribution: 40% beg, 40% int, 20% adv
    n_beg = math.ceil(count * 0.4)
    n_mid = math.ceil(count * 0.4)
    n_adv = count - n_beg - n_mid

    random.shuffle(beg)
    random.shuffle(mid)
    random.shuffle(adv)

    pool = beg[:n_beg] + mid[:n_mid] + adv[:max(n_adv, 0)]
    return pool


# ---------------------------------------------------------------------------
# Answer checking
# ---------------------------------------------------------------------------

def check_answer(question: dict, user_answer) -> bool:
    qtype = question["type"]
    if qtype == "mcq":
        return user_answer == question["correct_index"]
    elif qtype == "true_false":
        return user_answer == question["correct_answer"]
    elif qtype == "fill_blank":
        if user_answer is None:
            return False
        accepted = question.get("correct_answers", [])
        if not question.get("case_sensitive", False):
            accepted = [a.lower().strip() for a in accepted]
            return user_answer.lower().strip() in accepted
        return user_answer.strip() in accepted
    elif qtype == "code_snippet":
        return user_answer == question["correct_index"]
    return False


# ---------------------------------------------------------------------------
# XP computation
# ---------------------------------------------------------------------------

def compute_question_xp(
    question: dict,
    is_correct: bool,
    elapsed_seconds: float | None = None,
) -> int:
    if not is_correct:
        return 0
    diff = question.get("difficulty", "beginner")
    xp = XP_TABLE.get(diff, 10)
    if elapsed_seconds is not None and elapsed_seconds < SPEED_BONUS_SECONDS:
        xp += SPEED_BONUS_XP
    return xp


def compute_session_xp(
    questions: list[dict],
    results: dict,
    elapsed_per_question: dict | None = None,
    streak_days: int = 0,
) -> int:
    """
    Compute final session XP:
      - Sum raw XP per question (with speed bonuses if elapsed provided)
      - Apply streak multiplier
    """
    raw = 0
    for q in questions:
        elapsed = (elapsed_per_question or {}).get(q["id"])
        raw += compute_question_xp(q, results.get(q["id"], False), elapsed)
    mult = streak_multiplier(streak_days)
    return math.floor(raw * mult)
