from __future__ import annotations
import streamlit as st
from core.constants import TOPICS, DIFFICULTIES, TOPIC_LABELS, TOPIC_ICONS, DIFFICULTY_ICONS, BADGES


def render() -> None:
    st.title("My Progress")

    progress = st.session_state["progress"]
    topics = progress["topics"]
    sessions = progress.get("sessions", [])
    xp = progress["xp"]
    streak = progress["streak"]

    # --- Overall stats ---
    st.subheader("Overall Stats")
    total_attempted = sum(
        topics[t][d]["attempted"] for t in TOPICS for d in DIFFICULTIES
    )
    total_correct = sum(
        topics[t][d]["correct"] for t in TOPICS for d in DIFFICULTIES
    )
    overall_accuracy = total_correct / total_attempted if total_attempted else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total XP", f"{xp['total']:,}")
    c2.metric("Level", xp["level"])
    c3.metric("Questions Answered", total_attempted)
    c4.metric("Overall Accuracy", f"{overall_accuracy:.0%}")

    st.divider()

    # --- Per-topic breakdown ---
    st.subheader("Topics")
    for topic in TOPICS:
        t_stats = topics[topic]
        t_attempted = sum(t_stats[d]["attempted"] for d in DIFFICULTIES)
        t_correct = sum(t_stats[d]["correct"] for d in DIFFICULTIES)
        t_acc = t_correct / t_attempted if t_attempted else 0

        with st.expander(f"{TOPIC_ICONS[topic]} {TOPIC_LABELS[topic]} — {t_acc:.0%} accuracy ({t_attempted} answered)"):
            diff_data = {}
            for diff in DIFFICULTIES:
                s = t_stats[diff]
                acc = s["correct"] / s["attempted"] if s["attempted"] else 0
                diff_data[diff.title()] = acc

            if any(v > 0 for v in diff_data.values()):
                import pandas as pd
                df = pd.DataFrame.from_dict(
                    {k: [v] for k, v in diff_data.items()},
                    orient="columns",
                )
                st.bar_chart(df, height=150, y_label="Accuracy")

            cols = st.columns(3)
            for i, diff in enumerate(DIFFICULTIES):
                s = t_stats[diff]
                acc = f"{s['correct']/s['attempted']:.0%}" if s["attempted"] else "—"
                cols[i].metric(
                    f"{DIFFICULTY_ICONS[diff]} {diff.title()}",
                    acc,
                    f"{s['attempted']} attempted",
                )

    st.divider()

    # --- Badges ---
    st.subheader("Badges")
    earned = set(progress.get("badges", []))
    cols = st.columns(4)
    for i, (key, badge) in enumerate(BADGES.items()):
        col = cols[i % 4]
        if key in earned:
            col.markdown(f"**{badge['icon']} {badge['label']}**")
            col.caption(badge["desc"])
        else:
            col.markdown(f"🔒 ~~{badge['label']}~~")
            col.caption(badge["desc"])

    st.divider()

    # --- Recent sessions ---
    st.subheader("Recent Sessions")
    if not sessions:
        st.info("No sessions yet — take a quiz to see your history here!")
        return

    recent = list(reversed(sessions[-20:]))
    rows = []
    for s in recent:
        acc = s["questions_correct"] / s["questions_attempted"] if s["questions_attempted"] else 0
        rows.append({
            "Date": s["timestamp"][:10],
            "Topic": TOPIC_LABELS.get(s["topic"], s["topic"]),
            "Difficulty": s["difficulty"].title(),
            "Score": f"{s['questions_correct']}/{s['questions_attempted']}",
            "Accuracy": f"{acc:.0%}",
            "XP": s["xp_earned"],
        })

    import pandas as pd
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
