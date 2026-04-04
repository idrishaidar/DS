from __future__ import annotations
import json
import uuid
from datetime import date, datetime

from core.constants import (
    TOPICS, DIFFICULTIES, BADGES, STORAGE_KEY,
    xp_to_level, streak_multiplier,
)


# ---------------------------------------------------------------------------
# Default / init
# ---------------------------------------------------------------------------

def _default_topic_stats() -> dict:
    return {diff: {"attempted": 0, "correct": 0, "xp_earned": 0} for diff in DIFFICULTIES}


def default_progress() -> dict:
    return {
        "schema_version": "1.0",
        "created_at": datetime.utcnow().isoformat(),
        "last_active": datetime.utcnow().isoformat(),
        "xp": {"total": 0, "level": 1, "xp_this_level": 0, "xp_to_next_level": 100},
        "streak": {
            "current_days": 0,
            "longest_days": 0,
            "last_quiz_date": None,
            "grace_period_used_today": False,
        },
        "topics": {topic: _default_topic_stats() for topic in TOPICS},
        "question_history": {},
        "sessions": [],
        "badges": [],
    }


# ---------------------------------------------------------------------------
# Load / save via streamlit-local-storage
# ---------------------------------------------------------------------------

def load_progress(ls) -> dict:
    """Load progress from browser localStorage. Returns default if not found."""
    try:
        raw = ls.getItem(STORAGE_KEY)
        if raw:
            data = json.loads(raw) if isinstance(raw, str) else raw
            # Ensure all topics exist (handles schema additions over time)
            for topic in TOPICS:
                if topic not in data.get("topics", {}):
                    data.setdefault("topics", {})[topic] = _default_topic_stats()
            return data
    except Exception:
        pass
    return default_progress()


def save_progress(ls, progress: dict) -> None:
    """Persist progress to browser localStorage."""
    progress["last_active"] = datetime.utcnow().isoformat()
    ls.setItem(STORAGE_KEY, json.dumps(progress))


# ---------------------------------------------------------------------------
# Streak logic
# ---------------------------------------------------------------------------

def update_streak(progress: dict, today: date | None = None) -> dict:
    if today is None:
        today = date.today()
    streak = progress["streak"]
    last_raw = streak.get("last_quiz_date")

    if last_raw is None:
        streak["current_days"] = 1
        streak["longest_days"] = 1
    else:
        last = date.fromisoformat(last_raw)
        delta = (today - last).days
        if delta == 0:
            pass  # already updated today
        elif delta == 1:
            streak["current_days"] += 1
            streak["grace_period_used_today"] = False
        elif delta == 2 and not streak.get("grace_period_used_today", False):
            # One-time grace period
            streak["current_days"] += 1
            streak["grace_period_used_today"] = True
        else:
            streak["current_days"] = 1
            streak["grace_period_used_today"] = False

    streak["last_quiz_date"] = today.isoformat()
    streak["longest_days"] = max(streak["longest_days"], streak["current_days"])
    return progress


# ---------------------------------------------------------------------------
# XP helpers
# ---------------------------------------------------------------------------

def add_xp(progress: dict, xp: int) -> dict:
    progress["xp"]["total"] += xp
    level, xp_in, xp_needed = xp_to_level(progress["xp"]["total"])
    progress["xp"]["level"] = level
    progress["xp"]["xp_this_level"] = xp_in
    progress["xp"]["xp_to_next_level"] = xp_needed
    return progress


# ---------------------------------------------------------------------------
# Badge checks
# ---------------------------------------------------------------------------

def check_and_award_badges(progress: dict) -> list[str]:
    """Returns list of newly awarded badge keys."""
    earned = set(progress.get("badges", []))
    new_badges = []

    def award(key: str) -> None:
        if key not in earned:
            earned.add(key)
            new_badges.append(key)

    total_correct = sum(
        progress["topics"][t][d]["correct"]
        for t in TOPICS for d in DIFFICULTIES
    )
    sessions = progress.get("sessions", [])
    streak = progress["streak"]["current_days"]
    level = progress["xp"]["level"]

    if sessions:
        award("first_quiz")
    if streak >= 3:
        award("streak_3")
    if streak >= 7:
        award("streak_7")
    if streak >= 30:
        award("streak_30")
    if total_correct >= 100:
        award("century")
    if level >= 5:
        award("level_5")
    if level >= 10:
        award("level_10")
    if sessions:
        last = sessions[-1]
        if last["questions_attempted"] > 0 and last["questions_correct"] == last["questions_attempted"]:
            award("perfect_session")
    sql_adv_correct = progress["topics"].get("analytics_sql", {}).get("advanced", {}).get("correct", 0)
    if sql_adv_correct >= 20:
        award("sql_master")

    progress["badges"] = list(earned)
    return new_badges


# ---------------------------------------------------------------------------
# Post-session update (main entry point)
# ---------------------------------------------------------------------------

def update_after_session(
    progress: dict,
    topic: str,
    difficulty: str,
    questions: list[dict],
    answers: dict,          # {question_id: user_answer}
    results: dict,          # {question_id: bool}
    xp_earned: int,
    duration_seconds: int,
) -> tuple[dict, list[str]]:
    """
    Updates progress in-place after a completed quiz session.
    Returns (updated_progress, list_of_new_badge_keys).
    """
    session_id = str(uuid.uuid4())
    correct_count = sum(1 for v in results.values() if v)
    attempted = len(results)

    # Update topic stats
    if difficulty == "mixed":
        for q in questions:
            d = q["difficulty"]
            t = q["topic"]
            is_correct = results.get(q["id"], False)
            progress["topics"][t][d]["attempted"] += 1
            if is_correct:
                progress["topics"][t][d]["correct"] += 1
    else:
        stats = progress["topics"][topic][difficulty]
        stats["attempted"] += attempted
        stats["correct"] += correct_count

    # Update question history
    for q in questions:
        qid = q["id"]
        hist = progress["question_history"].setdefault(qid, {"seen": 0, "correct": 0, "last_seen": None})
        hist["seen"] += 1
        if results.get(qid, False):
            hist["correct"] += 1
        hist["last_seen"] = date.today().isoformat()

    # Record session
    session_record = {
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat(),
        "topic": topic,
        "difficulty": difficulty,
        "questions_attempted": attempted,
        "questions_correct": correct_count,
        "xp_earned": xp_earned,
        "duration_seconds": duration_seconds,
    }
    progress["sessions"].append(session_record)
    if len(progress["sessions"]) > 500:
        progress["sessions"] = progress["sessions"][-500:]

    # XP & streak
    add_xp(progress, xp_earned)
    update_streak(progress)

    # Badges
    new_badges = check_and_award_badges(progress)

    return progress, new_badges
