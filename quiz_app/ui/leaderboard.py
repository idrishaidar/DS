from __future__ import annotations
import streamlit as st
from core.constants import TOPIC_LABELS


def render() -> None:
    st.title("Leaderboard")
    st.caption("Your top sessions, ranked by XP earned.")

    sessions = st.session_state["progress"].get("sessions", [])
    if not sessions:
        st.info("No sessions yet — complete a quiz to appear here!")
        return

    # Sort by XP descending, show top 20
    sorted_sessions = sorted(sessions, key=lambda s: s["xp_earned"], reverse=True)[:20]

    rows = []
    for rank, s in enumerate(sorted_sessions, start=1):
        acc = s["questions_correct"] / s["questions_attempted"] if s["questions_attempted"] else 0
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")
        rows.append({
            "Rank": medal,
            "Date": s["timestamp"][:10],
            "Topic": TOPIC_LABELS.get(s["topic"], s["topic"]),
            "Difficulty": s["difficulty"].title(),
            "Score": f"{s['questions_correct']}/{s['questions_attempted']}",
            "Accuracy": f"{acc:.0%}",
            "XP Earned": s["xp_earned"],
        })

    import pandas as pd
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "XP Earned": st.column_config.NumberColumn(format="%d XP"),
        },
    )

    # Stats callout
    total_xp = sum(s["xp_earned"] for s in sessions)
    best_session = sorted_sessions[0] if sorted_sessions else None
    if best_session:
        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Sessions", len(sessions))
        col2.metric("All-time XP", f"{total_xp:,}")
        best_acc = best_session["questions_correct"] / best_session["questions_attempted"] if best_session["questions_attempted"] else 0
        col3.metric("Best Session XP", best_session["xp_earned"], f"{best_acc:.0%} accuracy")
