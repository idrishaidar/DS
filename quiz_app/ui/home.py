from __future__ import annotations
import streamlit as st
from core.constants import TOPICS, DIFFICULTIES, TOPIC_LABELS, TOPIC_ICONS
from ui.components import topic_card


def render(all_questions: dict) -> None:
    st.title("DS Interview Quiz")
    st.markdown(
        "Practice **Statistics**, **Machine Learning**, **Deep Learning**, "
        "**Analytics & SQL**, and **AI & LLMs** — Duolingo-style."
    )

    progress = st.session_state["progress"]

    # --- Topic grid ---
    st.subheader("Choose a Topic")
    for topic in TOPICS:
        topic_stats = progress["topics"][topic]
        clicked = topic_card(
            topic=topic,
            topic_stats=topic_stats,
            on_start_key=f"start_{topic}",
        )
        if clicked:
            st.session_state["selected_topic"] = topic
            st.session_state["screen"] = "quiz_config"
            st.rerun()


def render_quiz_config(all_questions: dict) -> None:
    topic = st.session_state.get("selected_topic", TOPICS[0])

    st.title(f"{TOPIC_ICONS[topic]} {TOPIC_LABELS[topic]} Quiz")

    if st.button("← Back to Home"):
        st.session_state["screen"] = "home"
        st.rerun()

    st.subheader("Configure your session")

    # Difficulty
    diff_options = ["Beginner", "Intermediate", "Advanced", "Mixed"]
    difficulty = st.radio(
        "Difficulty",
        diff_options,
        horizontal=True,
        index=0,
    ).lower()

    # Question count — limit to available questions
    if difficulty == "mixed":
        available = sum(
            len(all_questions.get(topic, {}).get(d, []))
            for d in DIFFICULTIES
        )
    else:
        available = len(all_questions.get(topic, {}).get(difficulty, []))

    max_count = min(available, 20)
    if max_count == 0:
        st.warning("No questions available for this selection. Run the generation script to populate the question bank.")
        return

    count_options = [c for c in [5, 10, 20] if c <= max_count]
    if not count_options:
        count_options = [max_count]

    count = st.radio(
        "Number of questions",
        count_options,
        horizontal=True,
        index=min(1, len(count_options) - 1),
    )

    st.divider()

    # XP preview
    from core.constants import XP_TABLE, streak_multiplier
    streak = st.session_state["progress"]["streak"]["current_days"]
    mult = streak_multiplier(streak)
    if difficulty == "mixed":
        avg_xp = 20
    else:
        avg_xp = XP_TABLE.get(difficulty, 10)
    max_xp = int(avg_xp * count * mult)
    st.info(f"Up to **{max_xp} XP** if you answer all {count} questions correctly  (streak multiplier: {mult}x)")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start!", type="primary", use_container_width=True):
            _start_quiz(topic, difficulty, count, all_questions)
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.session_state["screen"] = "home"
            st.rerun()


def _start_quiz(topic: str, difficulty: str, count: int, all_questions: dict) -> None:
    from core.quiz_engine import sample_questions
    from datetime import datetime

    history = st.session_state["progress"].get("question_history", {})
    questions = sample_questions(all_questions, topic, difficulty, count, history)

    if not questions:
        st.error("Could not load questions. Please check the question bank.")
        return

    st.session_state["quiz_state"] = {
        "topic": topic,
        "difficulty": difficulty,
        "questions": questions,
        "current_index": 0,
        "answers": {},
        "results": {},
        "elapsed": {},
        "start_time": datetime.utcnow().isoformat(),
        "question_start_time": datetime.utcnow().isoformat(),
        "submitted_current": False,
        "xp_earned": 0,
    }
    st.session_state["screen"] = "quiz"
    st.rerun()
