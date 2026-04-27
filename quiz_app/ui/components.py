from __future__ import annotations
import streamlit as st
from core.constants import TOPIC_ICONS, TOPIC_LABELS, TOPIC_DESCRIPTIONS, DIFFICULTY_ICONS, BADGES, MAX_LEVEL


def xp_bar(progress: dict) -> None:
    """Renders level + XP progress bar in sidebar."""
    xp = progress["xp"]
    level = xp["level"]
    xp_in = xp["xp_this_level"]
    xp_needed = xp["xp_to_next_level"]
    pct = min(xp_in / max(xp_needed, 1), 1.0)

    if level >= MAX_LEVEL:
        label = f"Level {level} (MAX) — {xp['total']} XP"
    else:
        label = f"Level {level} — {xp_in}/{xp_needed} XP"

    st.progress(pct, text=label)


def streak_badge(progress: dict) -> None:
    """Renders streak metric in sidebar."""
    streak = progress["streak"]["current_days"]
    longest = progress["streak"]["longest_days"]
    icon = "🔥" if streak >= 3 else ("✨" if streak >= 1 else "❄️")
    st.metric(
        label="Daily Streak",
        value=f"{icon} {streak} day{'s' if streak != 1 else ''}",
        delta=f"Best: {longest}d" if longest > 0 else None,
    )


def topic_card(
    topic: str,
    topic_stats: dict,
    on_start_key: str,
) -> bool:
    """
    Renders a bordered card for a topic. Returns True if the Start button was clicked.
    topic_stats: {difficulty: {attempted, correct, xp_earned}}
    """
    total_attempted = sum(s["attempted"] for s in topic_stats.values())
    total_correct = sum(s["correct"] for s in topic_stats.values())
    accuracy_str = f"{total_correct/total_attempted:.0%}" if total_attempted > 0 else "Not started"

    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### {TOPIC_ICONS[topic]} {TOPIC_LABELS[topic]}")
            st.caption(TOPIC_DESCRIPTIONS[topic])
        with col2:
            st.metric("Accuracy", accuracy_str)
            st.caption(f"{total_attempted} answered")

        # Per-difficulty breakdown
        diff_cols = st.columns(3)
        for i, diff in enumerate(["beginner", "intermediate", "advanced"]):
            s = topic_stats[diff]
            acc = f"{s['correct']/s['attempted']:.0%}" if s["attempted"] > 0 else "—"
            diff_cols[i].caption(f"{DIFFICULTY_ICONS[diff]} {diff.title()}: {acc}")

        clicked = st.button("Start Quiz", key=on_start_key, use_container_width=True)
    return clicked


def question_progress_bar(current: int, total: int) -> None:
    """Shows 'Question N of M' with a progress bar."""
    st.progress(current / total, text=f"Question {current} of {total}")


def answer_feedback(is_correct: bool, explanation: str) -> None:
    """Shows green/red feedback box with explanation."""
    if is_correct:
        st.success(f"Correct! {explanation}")
    else:
        st.error(f"Incorrect. {explanation}")


def badge_display(badge_keys: list[str]) -> None:
    """Shows a row of earned badges."""
    if not badge_keys:
        st.caption("No badges yet — complete quizzes to earn them!")
        return
    cols = st.columns(min(len(badge_keys), 4))
    for i, key in enumerate(badge_keys):
        badge = BADGES.get(key, {"icon": "🏅", "label": key, "desc": ""})
        with cols[i % 4]:
            st.markdown(f"**{badge['icon']} {badge['label']}**")
            st.caption(badge["desc"])


def new_badges_popup(new_badge_keys: list[str]) -> None:
    """Shows a toast/expander for newly earned badges."""
    if not new_badge_keys:
        return
    for key in new_badge_keys:
        badge = BADGES.get(key, {"icon": "🏅", "label": key})
        st.toast(f"{badge['icon']} Badge unlocked: **{badge['label']}**!", icon="🎉")
