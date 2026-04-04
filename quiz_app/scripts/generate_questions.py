#!/usr/bin/env python3
"""
One-time script to generate quiz question banks using Claude API.
Run from quiz_app/ directory:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-...
    python scripts/generate_questions.py

Idempotent: skips files that already exist (use --force to regenerate).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import anthropic

DATA_DIR = Path(__file__).parent.parent / "data"

TOPICS = ["statistics", "machine_learning", "deep_learning", "analytics_sql", "ai_llms"]
DIFFICULTIES = ["beginner", "intermediate", "advanced"]

TOPIC_CONTEXT = {
    "statistics": (
        "probability theory, descriptive statistics, distributions (normal, binomial, Poisson, t, chi-squared, F), "
        "hypothesis testing, p-values, confidence intervals, Bayesian inference, correlation vs causation, "
        "sampling methods, central limit theorem, law of large numbers, effect size, statistical power, "
        "Type I/II errors, A/B testing, regression fundamentals, ANOVA"
    ),
    "machine_learning": (
        "supervised learning (linear/logistic regression, decision trees, random forests, gradient boosting, SVMs, KNN), "
        "unsupervised learning (k-means, DBSCAN, PCA, t-SNE, autoencoders), "
        "model evaluation (accuracy, precision, recall, F1, AUC-ROC, confusion matrix, cross-validation), "
        "regularization (L1/L2/elastic net, dropout), bias-variance tradeoff, overfitting, "
        "feature engineering, feature selection, imbalanced datasets (SMOTE, class weights), "
        "hyperparameter tuning, ensemble methods, stacking, boosting vs bagging"
    ),
    "deep_learning": (
        "neural network fundamentals (activation functions, backpropagation, vanishing/exploding gradients), "
        "optimizers (SGD, Adam, RMSProp, AdaGrad), batch normalization, layer normalization, "
        "CNNs (convolution, pooling, receptive field, stride, padding, architectures: ResNet, VGG, EfficientNet), "
        "RNNs, LSTMs, GRUs, sequence-to-sequence, attention mechanisms, "
        "Transformer architecture (self-attention, multi-head attention, positional encoding), "
        "transfer learning, fine-tuning, dropout, weight initialization, learning rate schedules"
    ),
    "analytics_sql": (
        "SQL fundamentals (SELECT, WHERE, GROUP BY, HAVING, ORDER BY, LIMIT), "
        "JOINs (INNER, LEFT, RIGHT, FULL, CROSS, SELF), subqueries, CTEs (WITH), "
        "window functions (ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD, SUM OVER, AVG OVER, PARTITION BY), "
        "aggregation functions, CASE WHEN, COALESCE, NULL handling, "
        "data types, indexing, query optimisation, EXPLAIN, "
        "data modelling (star schema, snowflake, normalisation, denormalisation), "
        "Python analytics with pandas, numpy aggregations, data cleaning, reshaping (melt, pivot)"
    ),
    "ai_llms": (
        "large language model fundamentals (tokenisation, embeddings, temperature, top-p, top-k), "
        "GPT family, BERT, T5, LLaMA architecture differences, "
        "RLHF (reinforcement learning from human feedback), PPO, DPO, "
        "prompt engineering (zero-shot, few-shot, chain-of-thought, ReAct), "
        "RAG (retrieval-augmented generation), vector databases, embedding similarity, "
        "AI agents, tool use, function calling, "
        "hallucinations, grounding, evaluation of LLMs (BLEU, ROUGE, perplexity, human eval), "
        "fine-tuning LLMs (LoRA, QLoRA, PEFT), quantisation, inference optimisation, "
        "responsible AI, bias in models, safety, alignment"
    ),
}

DIFFICULTY_GUIDANCE = {
    "beginner": (
        "Suitable for someone with 0-6 months of exposure to the field. "
        "Focus on definitions, basic concepts, and intuitive understanding. "
        "True/False and simple MCQ work well here."
    ),
    "intermediate": (
        "Suitable for someone actively studying or with 1-2 years experience. "
        "Include conceptual traps, formula recall, and reasoning questions. "
        "Code snippets with Python/SQL completions work well."
    ),
    "advanced": (
        "Suitable for senior practitioners or FAANG interview prep. "
        "Questions should require deep understanding, mathematical nuance, or subtle edge cases. "
        "Code snippets and fill-in-the-blank with precise technical answers work well."
    ),
}

SCHEMA_EXAMPLE = """
JSON schema for each question type. You MUST follow this exactly.

Common fields (all types):
  "id": string, format "{topic_abbrev}_{diff_abbrev}_{3-digit-seq}" e.g. "ml_int_042"
  "type": one of "mcq" | "true_false" | "fill_blank" | "code_snippet"
  "topic": the topic slug (e.g. "machine_learning")
  "difficulty": the difficulty slug (e.g. "intermediate")
  "tags": list of 2-4 keyword strings
  "explanation": 1-3 sentence explanation shown after answering

MCQ example:
{
  "id": "ml_int_001",
  "type": "mcq",
  "topic": "machine_learning",
  "difficulty": "intermediate",
  "question": "Which regularization technique can set feature weights exactly to zero?",
  "options": ["Ridge (L2)", "Lasso (L1)", "Elastic Net", "Weight decay"],
  "correct_index": 1,
  "explanation": "Lasso (L1) regularization adds a penalty proportional to the absolute value of weights, which geometrically touches the coordinate axes and can produce exactly zero weights, yielding sparse models.",
  "tags": ["regularization", "lasso", "feature-selection"]
}

True/False example:
{
  "id": "stat_beg_007",
  "type": "true_false",
  "topic": "statistics",
  "difficulty": "beginner",
  "question": "A p-value of 0.03 means there is a 3% probability the null hypothesis is true.",
  "correct_answer": false,
  "explanation": "A p-value is the probability of observing results at least as extreme as the sample data, assuming H0 is true — not the probability that H0 itself is true.",
  "tags": ["p-value", "hypothesis-testing", "common-misconception"]
}

Fill-in-the-blank example:
{
  "id": "stat_int_018",
  "type": "fill_blank",
  "topic": "statistics",
  "difficulty": "intermediate",
  "question": "The central limit theorem states that the sampling distribution of the mean approaches a ______ distribution as sample size increases, regardless of the population distribution.",
  "blank_marker": "______",
  "correct_answers": ["normal", "gaussian", "bell curve", "normal distribution"],
  "case_sensitive": false,
  "explanation": "The CLT guarantees that the sampling distribution of the mean converges to a normal distribution for sufficiently large n (typically n ≥ 30), given finite mean and variance.",
  "tags": ["CLT", "sampling-distribution", "normal-distribution"]
}

Code snippet example (SQL):
{
  "id": "sql_adv_015",
  "type": "code_snippet",
  "topic": "analytics_sql",
  "difficulty": "advanced",
  "language": "sql",
  "preamble": "Given a table orders(order_id, customer_id, amount), which keyword completes this query to return the top 3 customers by total spend?",
  "code_block": "SELECT customer_id, SUM(amount) AS total\\nFROM orders\\nGROUP BY customer_id\\nORDER BY total ____\\nLIMIT 3;",
  "blank_marker": "____",
  "options": ["ASC", "DESC", "TOTAL DESC", "ASCENDING"],
  "correct_index": 1,
  "explanation": "ORDER BY total DESC sorts highest spend first. Without DESC, ASC is the default, which would return the 3 lowest spenders.",
  "tags": ["ORDER BY", "aggregation", "LIMIT", "TOP-N"]
}

Code snippet example (Python):
{
  "id": "ml_adv_011",
  "type": "code_snippet",
  "topic": "machine_learning",
  "difficulty": "advanced",
  "language": "python",
  "preamble": "What does this scikit-learn pipeline do when fit() is called?",
  "code_block": "from sklearn.pipeline import Pipeline\\nfrom sklearn.preprocessing import StandardScaler\\nfrom sklearn.linear_model import LogisticRegression\\n\\npipe = Pipeline([\\n    ('scaler', StandardScaler()),\\n    ('clf', LogisticRegression())\\n])\\npipe.fit(X_train, y_train)",
  "blank_marker": null,
  "options": [
    "Scales X_train then fits logistic regression on scaled data",
    "Fits logistic regression first, then scales the coefficients",
    "Scales X_train and y_train independently",
    "Only scales X_train; fitting is deferred until predict() is called"
  ],
  "correct_index": 0,
  "explanation": "A sklearn Pipeline applies transforms sequentially before the final estimator. fit() calls fit_transform() on the scaler, then fit() on the classifier using the scaled features.",
  "tags": ["pipeline", "preprocessing", "sklearn", "StandardScaler"]
}
"""


def build_prompt(topic: str, difficulty: str, counts: dict[str, int]) -> str:
    topic_abbrev = {
        "statistics": "stat",
        "machine_learning": "ml",
        "deep_learning": "dl",
        "analytics_sql": "sql",
        "ai_llms": "ai",
    }[topic]
    diff_abbrev = {"beginner": "beg", "intermediate": "int", "advanced": "adv"}[difficulty]
    total = sum(counts.values())

    return f"""You are an expert data scientist and educator creating quiz questions for a Duolingo-style interview prep app.

## Task
Generate exactly {total} quiz questions for:
- Topic: {topic} ({TOPIC_CONTEXT[topic]})
- Difficulty: {difficulty} — {DIFFICULTY_GUIDANCE[difficulty]}

## Question type distribution
- MCQ (multiple choice, 4 options): {counts['mcq']} questions
- True/False: {counts['true_false']} questions
- Fill-in-the-blank: {counts['fill_blank']} questions
- Code snippet (Python or SQL completions): {counts['code_snippet']} questions

## Schema to follow
{SCHEMA_EXAMPLE}

## ID format for this batch
Use prefix "{topic_abbrev}_{diff_abbrev}_" followed by a zero-padded 3-digit sequence: 001, 002, 003, ...
Example: "{topic_abbrev}_{diff_abbrev}_001", "{topic_abbrev}_{diff_abbrev}_002", etc.

## Quality requirements
1. Every question must have a clear, unambiguous correct answer
2. Distractors (wrong MCQ options) must be plausible but clearly wrong to an expert
3. Explanations must be accurate and educational (1-3 sentences)
4. Avoid trivially easy true/false that anyone could guess
5. Fill-in-the-blank answers should accept common synonyms in correct_answers list
6. Code snippets must be syntactically correct and the correct_index must match an option in the options list
7. Questions must cover a diverse range of sub-topics within {topic}
8. Do NOT repeat questions — each must test a distinct concept

## Output format
Return ONLY a valid JSON object with this structure — no markdown fences, no commentary:
{{
  "topic": "{topic}",
  "difficulty": "{difficulty}",
  "version": "1.0",
  "questions": [ ... all {total} questions ... ]
}}"""


def question_type_counts(total: int) -> dict[str, int]:
    """Distribute questions across types: 50% MCQ, 20% T/F, 20% fill_blank, 10% code."""
    n_mcq = round(total * 0.50)
    n_tf = round(total * 0.20)
    n_fill = round(total * 0.20)
    n_code = total - n_mcq - n_tf - n_fill
    return {"mcq": n_mcq, "true_false": n_tf, "fill_blank": n_fill, "code_snippet": max(n_code, 1)}


def generate_bank(client: anthropic.Anthropic, topic: str, difficulty: str, total: int) -> dict:
    counts = question_type_counts(total)
    prompt = build_prompt(topic, difficulty, counts)
    print(f"  Calling Claude API ({total} questions, types: {counts})...")
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    # Strip markdown fences if the model wrapped the JSON anyway
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]
    return json.loads(raw)


def validate_bank(bank: dict, topic: str, difficulty: str) -> list[str]:
    """Returns list of validation errors."""
    errors = []
    questions = bank.get("questions", [])
    if not questions:
        errors.append("No questions found")
        return errors
    seen_ids: set[str] = set()
    for i, q in enumerate(questions):
        prefix = f"Q{i+1} (id={q.get('id','?')})"
        for field in ("id", "type", "topic", "difficulty", "question", "explanation"):
            if field not in q:
                errors.append(f"{prefix}: missing '{field}'")
        if q.get("id") in seen_ids:
            errors.append(f"{prefix}: duplicate id")
        seen_ids.add(q.get("id", f"__missing_{i}"))
        qtype = q.get("type")
        if qtype == "mcq":
            if not q.get("options") or len(q["options"]) != 4:
                errors.append(f"{prefix}: MCQ needs exactly 4 options")
            if not isinstance(q.get("correct_index"), int):
                errors.append(f"{prefix}: MCQ needs integer correct_index")
        elif qtype == "true_false":
            if not isinstance(q.get("correct_answer"), bool):
                errors.append(f"{prefix}: true_false needs boolean correct_answer")
        elif qtype == "fill_blank":
            if not q.get("correct_answers"):
                errors.append(f"{prefix}: fill_blank needs correct_answers list")
        elif qtype == "code_snippet":
            if not q.get("options") or not isinstance(q.get("correct_index"), int):
                errors.append(f"{prefix}: code_snippet needs options + correct_index")
    return errors


QUESTION_COUNTS = {
    "statistics":       {"beginner": 25, "intermediate": 25, "advanced": 20},
    "machine_learning": {"beginner": 25, "intermediate": 30, "advanced": 20},
    "deep_learning":    {"beginner": 20, "intermediate": 25, "advanced": 20},
    "analytics_sql":    {"beginner": 20, "intermediate": 25, "advanced": 20},
    "ai_llms":          {"beginner": 15, "intermediate": 20, "advanced": 15},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DS quiz question banks via Claude API")
    parser.add_argument("--force", action="store_true", help="Regenerate even if file already exists")
    parser.add_argument("--topic", help="Only generate for this topic")
    parser.add_argument("--difficulty", help="Only generate for this difficulty")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    topics = [args.topic] if args.topic else TOPICS
    difficulties = [args.difficulty] if args.difficulty else DIFFICULTIES

    total_generated = 0
    total_skipped = 0
    total_errors = 0

    for topic in topics:
        for difficulty in difficulties:
            out_path = DATA_DIR / topic / f"{difficulty}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if out_path.exists() and not args.force:
                print(f"[SKIP]  {topic}/{difficulty}.json already exists (use --force to regenerate)")
                total_skipped += 1
                continue

            count = QUESTION_COUNTS[topic][difficulty]
            print(f"[GEN]   {topic}/{difficulty} — {count} questions")

            max_retries = 3
            bank = None
            for attempt in range(1, max_retries + 1):
                try:
                    bank = generate_bank(client, topic, difficulty, count)
                    errors = validate_bank(bank, topic, difficulty)
                    if errors:
                        print(f"  Validation errors (attempt {attempt}):")
                        for e in errors[:5]:
                            print(f"    - {e}")
                        if attempt < max_retries:
                            print("  Retrying...")
                            time.sleep(2 ** attempt)
                            bank = None
                            continue
                        else:
                            print("  WARNING: Saving despite validation errors")
                    break
                except json.JSONDecodeError as e:
                    print(f"  JSON parse error (attempt {attempt}): {e}")
                    if attempt < max_retries:
                        time.sleep(2 ** attempt)
                    else:
                        print(f"  FAILED after {max_retries} attempts — skipping")
                        total_errors += 1
                        bank = None
                except Exception as e:
                    print(f"  API error (attempt {attempt}): {e}")
                    if attempt < max_retries:
                        time.sleep(2 ** attempt)
                    else:
                        print(f"  FAILED — skipping")
                        total_errors += 1
                        bank = None

            if bank is not None:
                with open(out_path, "w") as f:
                    json.dump(bank, f, indent=2)
                q_count = len(bank.get("questions", []))
                print(f"  ✓ Saved {q_count} questions to {out_path.relative_to(DATA_DIR.parent)}")
                total_generated += 1
            # Small delay to avoid rate limits
            time.sleep(1)

    print(f"\nDone. Generated: {total_generated}, Skipped: {total_skipped}, Errors: {total_errors}")


if __name__ == "__main__":
    main()
