from __future__ import annotations

TOPICS = ["statistics", "machine_learning", "deep_learning", "analytics_sql", "ai_llms"]
DIFFICULTIES = ["beginner", "intermediate", "advanced"]

TOPIC_LABELS = {
    "statistics": "Statistics",
    "machine_learning": "Machine Learning",
    "deep_learning": "Deep Learning",
    "analytics_sql": "Analytics & SQL",
    "ai_llms": "AI & LLMs",
}

TOPIC_ICONS = {
    "statistics": "📊",
    "machine_learning": "🤖",
    "deep_learning": "🧠",
    "analytics_sql": "🗄️",
    "ai_llms": "✨",
}

TOPIC_DESCRIPTIONS = {
    "statistics": "Probability, distributions, hypothesis testing, Bayesian methods",
    "machine_learning": "Supervised/unsupervised learning, metrics, feature engineering",
    "deep_learning": "Neural networks, CNNs, RNNs, Transformers, training techniques",
    "analytics_sql": "SQL queries, window functions, aggregations, data modeling",
    "ai_llms": "Large language models, RAG, agents, prompt engineering, RLHF",
}

DIFFICULTY_LABELS = {
    "beginner": "Beginner",
    "intermediate": "Intermediate",
    "advanced": "Advanced",
}

DIFFICULTY_ICONS = {
    "beginner": "🟢",
    "intermediate": "🟡",
    "advanced": "🔴",
}

# XP awarded per correct answer
XP_TABLE = {
    "beginner": 10,
    "intermediate": 20,
    "advanced": 40,
}

SPEED_BONUS_XP = 5
SPEED_BONUS_SECONDS = 10

# Streak multiplier: (min_days, multiplier)
STREAK_MULTIPLIERS = [
    (30, 2.0),
    (14, 1.5),
    (7, 1.25),
    (3, 1.1),
    (0, 1.0),
]

# XP required to REACH each level (index = level - 1)
LEVEL_THRESHOLDS = [0, 100, 250, 450, 700, 1050, 1500, 2100, 2850, 3850, 5350]
MAX_LEVEL = 10


def xp_to_level(total_xp: int) -> tuple[int, int, int]:
    """Returns (level, xp_earned_this_level, xp_needed_for_next_level)."""
    level = 1
    for i in range(len(LEVEL_THRESHOLDS) - 1, -1, -1):
        if total_xp >= LEVEL_THRESHOLDS[i]:
            level = min(i + 1, MAX_LEVEL)
            break
    xp_in = total_xp - LEVEL_THRESHOLDS[level - 1]
    if level < MAX_LEVEL:
        xp_needed = LEVEL_THRESHOLDS[level] - LEVEL_THRESHOLDS[level - 1]
    else:
        xp_needed = LEVEL_THRESHOLDS[-1] - LEVEL_THRESHOLDS[-2]
    return level, xp_in, xp_needed


def streak_multiplier(streak_days: int) -> float:
    for min_days, mult in STREAK_MULTIPLIERS:
        if streak_days >= min_days:
            return mult
    return 1.0


BADGES = {
    "first_quiz":      {"label": "First Step",    "icon": "🎯", "desc": "Complete your first quiz"},
    "streak_3":        {"label": "On a Roll",      "icon": "🔥", "desc": "3-day streak"},
    "streak_7":        {"label": "Week Warrior",   "icon": "⚡", "desc": "7-day streak"},
    "streak_30":       {"label": "Unstoppable",    "icon": "🏆", "desc": "30-day streak"},
    "perfect_session": {"label": "Flawless",       "icon": "💎", "desc": "100% accuracy in a session"},
    "century":         {"label": "Centurion",      "icon": "💯", "desc": "Answer 100 questions correctly"},
    "sql_master":      {"label": "SQL Wizard",     "icon": "🗄️", "desc": "20 correct advanced SQL answers"},
    "level_5":         {"label": "Rising Star",    "icon": "⭐", "desc": "Reach level 5"},
    "level_10":        {"label": "Data Guru",      "icon": "🧙", "desc": "Reach level 10"},
}

# localStorage key for progress data
STORAGE_KEY = "ds_quiz_progress"

# Question counts used in the generation script
QUESTION_COUNTS = {
    "statistics":       {"beginner": 25, "intermediate": 25, "advanced": 20},
    "machine_learning": {"beginner": 25, "intermediate": 30, "advanced": 20},
    "deep_learning":    {"beginner": 20, "intermediate": 25, "advanced": 20},
    "analytics_sql":    {"beginner": 20, "intermediate": 25, "advanced": 20},
    "ai_llms":          {"beginner": 15, "intermediate": 20, "advanced": 15},
}
