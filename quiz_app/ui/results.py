from __future__ import annotations
from datetime import datetime
import streamlit as st
from core.quiz_engine import compute_session_xp
from core.progress_manager import update_after_session, save_progress
from core.constants import TOPIC_ICONS, TOPIC_LABELS
from ui.components import new_badges_popup


def render(ls) -> None:
    """Results screen shown after a quiz is completed."""
    qs = st.session_state.get("quiz_state")
    if not qs:
        st.session_state["screen"] = "home"
        st.rerun()
        return

    # --- Compute final XP (with streak multiplier) ---
    questions = qs["questions"]
    results = qs["results"]
    elapsed = qs.get("elapsed", {})
    streak = st.session_state["progress"]["streak"]["current_days"]

    final_xp = compute_session_xp(questions, results, elapsed, streak)

    # --- Persist progress (only once) ---
    if not st.session_state.get("_results_saved"):
        start_str = qs.get("start_time")
        duration = 0
        if start_str:
            try:
                duration = int((datetime.utcnow() - datetime.fromisoformat(start_str)).total_seconds())
            except Exception:
                pass

        progress, new_badges = update_after_session(
            st.session_state["progress"],
            topic=qs["topic"],
            difficulty=qs["difficulty"],
            questions=questions,
            answers=qs["answers"],
            results=results,
            xp_earned=final_xp,
            duration_seconds=duration,
        )
        st.session_state["progress"] = progress
        st.session_state["_new_badges"] = new_badges
        save_progress(ls, progress)
        st.session_state["_results_saved"] = True

    new_badges = st.session_state.get("_new_badges", [])

    # --- UI ---
    topic = qs["topic"]
    correct = sum(1 for v in results.values() if v)
    total = len(questions)
    accuracy = correct / total if total > 0 else 0

    st.title("Quiz Complete!")

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Score", f"{correct}/{total}", f"{accuracy:.0%}")
    col2.metric("XP Earned", f"+{final_xp} XP")
    streak_now = st.session_state["progress"]["streak"]["current_days"]
    col3.metric("Streak", f"🔥 {streak_now} day{'s' if streak_now != 1 else ''}")

    if accuracy == 1.0:
        st.balloons()
        st.success("Perfect score! Amazing work!")
    elif accuracy >= 0.7:
        st.success("Great job! Keep it up!")
    else:
        st.info("Keep practising — you'll get there!")

    # Badge notifications
    new_badges_popup(new_badges)

    st.divider()

    # --- Per-question review accordion ---
    st.subheader("Question Review")
    for i, q in enumerate(questions):
        is_correct = results.get(q["id"], False)
        icon = "✅" if is_correct else "❌"
        qtype_label = {"mcq": "MCQ", "true_false": "T/F", "fill_blank": "Fill", "code_snippet": "Code"}.get(q["type"], q["type"])

        with st.expander(f"{icon} Q{i+1} [{qtype_label}] — {q['question'][:80]}..."):
            # Show question
            if q["type"] == "code_snippet":
                st.write(q.get("preamble", ""))
                st.code(q.get("code_block", ""), language=q.get("language", "python"))
            else:
                st.write(q["question"])

            # Show user's answer vs correct
            user_ans = qs["answers"].get(q["id"])
            if q["type"] in ("mcq", "code_snippet"):
                correct_label = q["options"][q["correct_index"]]
                user_label = q["options"][user_ans] if isinstance(user_ans, int) and user_ans < len(q["options"]) else str(user_ans)
                if not is_correct:
                    st.markdown(f"Your answer: **{user_label}**")
                st.markdown(f"Correct answer: **{correct_label}**")
            elif q["type"] == "true_false":
                st.markdown(f"Correct answer: **{q['correct_answer']}**")
                if not is_correct:
                    st.markdown(f"Your answer: **{user_ans}**")
            elif q["type"] == "fill_blank":
                st.markdown(f"Accepted: **{', '.join(q['correct_answers'])}**")
                if not is_correct:
                    st.markdown(f"Your answer: **{user_ans}**")

            # Explanation
            st.info(q.get("explanation", ""))

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Home", use_container_width=True):
            # Clear quiz state
            for key in ("quiz_state", "_results_saved", "_new_badges"):
                st.session_state.pop(key, None)
            st.session_state["screen"] = "home"
            st.rerun()
    with col2:
        if st.button("Play Again", type="primary", use_container_width=True):
            for key in ("quiz_state", "_results_saved", "_new_badges"):
                st.session_state.pop(key, None)
            st.session_state["screen"] = "quiz_config"
            st.rerun()
