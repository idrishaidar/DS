from __future__ import annotations
from datetime import datetime
import streamlit as st
from core.quiz_engine import check_answer, compute_question_xp
from ui.components import question_progress_bar, answer_feedback


def render() -> None:
    qs = st.session_state.get("quiz_state")
    if not qs:
        st.session_state["screen"] = "home"
        st.rerun()
        return

    questions = qs["questions"]
    idx = qs["current_index"]
    total = len(questions)

    # Quiz complete — transition to results
    if idx >= total:
        st.session_state["screen"] = "results"
        st.rerun()
        return

    q = questions[idx]

    # Header
    from core.constants import TOPIC_ICONS, TOPIC_LABELS, DIFFICULTY_ICONS
    topic = qs["topic"]
    diff = q.get("difficulty", qs["difficulty"])
    st.markdown(
        f"**{TOPIC_ICONS.get(topic, '')} {TOPIC_LABELS.get(topic, topic)}** "
        f"· {DIFFICULTY_ICONS.get(diff, '')} {diff.title()}"
    )
    question_progress_bar(idx + 1, total)

    already_submitted = qs.get("submitted_current", False)

    if not already_submitted:
        _render_question(q, qs)
    else:
        # Show result feedback + Next button
        _render_feedback(q, qs, idx, total)


def _render_question(q: dict, qs: dict) -> None:
    qtype = q["type"]
    form_key = f"q_form_{q['id']}"

    with st.form(form_key, clear_on_submit=False):
        if qtype == "mcq":
            st.subheader(q["question"])
            user_input = st.radio("Select one:", q["options"], index=None)
        elif qtype == "true_false":
            st.subheader(q["question"])
            user_input = st.radio("True or False?", ["True", "False"], index=None)
        elif qtype == "fill_blank":
            display = q["question"].replace(
                q.get("blank_marker", "______"),
                f":blue[**{'_' * 8}**]",
            )
            st.subheader(display)
            user_input = st.text_input("Your answer:", placeholder="Type your answer here")
        elif qtype == "code_snippet":
            st.subheader(q.get("preamble", "What does this code do?"))
            lang = q.get("language", "python")
            code = q.get("code_block", "")
            if q.get("blank_marker") and q["blank_marker"] in code:
                # Highlight blank with comment
                code = code.replace(q["blank_marker"], "/* ??? */")
            st.code(code, language=lang)
            user_input = st.radio("Select the correct answer:", q["options"], index=None)
        else:
            st.error(f"Unknown question type: {qtype}")
            user_input = None

        submitted = st.form_submit_button("Submit Answer", type="primary", use_container_width=True)

    if submitted:
        _handle_submit(q, qs, user_input)


def _handle_submit(q: dict, qs: dict, user_input) -> None:
    qtype = q["type"]

    # Normalise answer
    if qtype == "mcq":
        if user_input is None:
            st.warning("Please select an answer before submitting.")
            return
        answer = q["options"].index(user_input)
    elif qtype == "true_false":
        if user_input is None:
            st.warning("Please select True or False.")
            return
        answer = (user_input == "True")
    elif qtype == "fill_blank":
        answer = user_input or ""
    elif qtype == "code_snippet":
        if user_input is None:
            st.warning("Please select an answer before submitting.")
            return
        answer = q["options"].index(user_input)
    else:
        answer = user_input

    # Compute elapsed time
    start_str = qs.get("question_start_time")
    elapsed = None
    if start_str:
        try:
            elapsed = (datetime.utcnow() - datetime.fromisoformat(start_str)).total_seconds()
        except Exception:
            pass

    is_correct = check_answer(q, answer)
    xp = compute_question_xp(q, is_correct, elapsed)

    qs["answers"][q["id"]] = answer
    qs["results"][q["id"]] = is_correct
    qs["elapsed"][q["id"]] = elapsed
    qs["xp_earned"] += xp
    qs["submitted_current"] = True
    st.rerun()


def _render_feedback(q: dict, qs: dict, idx: int, total: int) -> None:
    is_correct = qs["results"].get(q["id"], False)
    xp_gained = 0
    if is_correct:
        from core.quiz_engine import compute_question_xp
        xp_gained = compute_question_xp(q, True, qs["elapsed"].get(q["id"]))

    # Show question again (read-only)
    qtype = q["type"]
    if qtype == "code_snippet":
        st.subheader(q.get("preamble", ""))
        st.code(q.get("code_block", ""), language=q.get("language", "python"))
    else:
        st.subheader(q["question"])

    # Show correct answer
    if qtype == "mcq" or qtype == "code_snippet":
        correct_label = q["options"][q["correct_index"]]
        user_ans_idx = qs["answers"].get(q["id"])
        if not is_correct and user_ans_idx is not None:
            st.markdown(f"Your answer: ~~{q['options'][user_ans_idx]}~~")
        st.markdown(f"Correct answer: **{correct_label}**")
    elif qtype == "true_false":
        st.markdown(f"Correct answer: **{q['correct_answer']}**")
    elif qtype == "fill_blank":
        st.markdown(f"Accepted answers: **{', '.join(q['correct_answers'])}**")

    # Feedback box
    answer_feedback(is_correct, q.get("explanation", ""))

    if is_correct and xp_gained:
        st.success(f"+{xp_gained} XP")

    # Next / Finish button
    is_last = idx + 1 >= total
    btn_label = "Finish Quiz" if is_last else "Next Question"

    if st.button(btn_label, type="primary", use_container_width=True):
        qs["current_index"] += 1
        qs["submitted_current"] = False
        qs["question_start_time"] = datetime.utcnow().isoformat()
        st.rerun()
