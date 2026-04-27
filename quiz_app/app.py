"""
DS Interview Quiz — Duolingo-style quiz app for data science interview prep.

Run: streamlit run app.py
"""
from __future__ import annotations
import streamlit as st
from streamlit_local_storage import LocalStorage

from core.progress_manager import load_progress
from core.question_loader import load_all_questions
from core.constants import TOPIC_ICONS, TOPIC_LABELS
from ui.components import xp_bar, streak_badge
from ui import home, quiz, results, progress, leaderboard

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DS Interview Quiz",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Bootstrap: localStorage, progress, questions
# ---------------------------------------------------------------------------
ls = LocalStorage()

if "screen" not in st.session_state:
    st.session_state["screen"] = "home"

if "progress" not in st.session_state:
    st.session_state["progress"] = load_progress(ls)

if "all_questions" not in st.session_state:
    with st.spinner("Loading question bank..."):
        st.session_state["all_questions"] = load_all_questions()

all_questions = st.session_state["all_questions"]
prog = st.session_state["progress"]

# ---------------------------------------------------------------------------
# Sidebar (persistent)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎯 DS Quiz")
    st.divider()

    streak_badge(prog)
    xp_bar(prog)

    st.divider()
    st.markdown("**Navigate**")

    if st.button("🏠 Home", use_container_width=True):
        st.session_state["screen"] = "home"
        st.rerun()

    if st.button("📈 My Progress", use_container_width=True):
        st.session_state["screen"] = "progress_screen"
        st.rerun()

    if st.button("🏆 Leaderboard", use_container_width=True):
        st.session_state["screen"] = "leaderboard"
        st.rerun()

    st.divider()

    # Quick-start buttons per topic
    st.markdown("**Quick Start**")
    from core.constants import TOPICS
    for topic in TOPICS:
        if st.button(
            f"{TOPIC_ICONS[topic]} {TOPIC_LABELS[topic]}",
            key=f"sidebar_{topic}",
            use_container_width=True,
        ):
            st.session_state["selected_topic"] = topic
            st.session_state["screen"] = "quiz_config"
            st.rerun()

# ---------------------------------------------------------------------------
# Screen router
# ---------------------------------------------------------------------------
screen = st.session_state["screen"]

if screen == "home":
    home.render(all_questions)

elif screen == "quiz_config":
    home.render_quiz_config(all_questions)

elif screen == "quiz":
    quiz.render()

elif screen == "results":
    results.render(ls)

elif screen == "progress_screen":
    progress.render()

elif screen == "leaderboard":
    leaderboard.render()

else:
    st.session_state["screen"] = "home"
    st.rerun()
