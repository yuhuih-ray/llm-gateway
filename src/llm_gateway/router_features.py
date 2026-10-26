"""Question-only numeric features shared by offline training and future serving."""

import re

FEATURE_NAMES = (
    "characters",
    "words",
    "sentences",
    "digits",
    "numbers",
    "math_operators",
    "latex_markers",
    "code_markers",
    "question_marks",
    "mean_word_length",
    "uppercase_ratio",
    "calculate",
    "compute",
    "how_many",
    "explain",
    "why",
    "which",
)


def extract_features(
    question: str,
) -> list[float]:  # Raw question, never a prompt or metadata.
    words = re.findall(r"\b\w+\b", question)
    letters = [char for char in question if char.isalpha()]
    values = [
        len(question),
        len(words),
        len(re.findall(r"[^.!?]+(?:[.!?]+|$)", question.strip())),
        sum(char.isdigit() for char in question),
        len(re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", question)),
        len(re.findall(r"[+*/=<>−×÷-]", question)),
        len(re.findall(r"\$|\\[A-Za-z]+|\\[()\[\]]", question)),
        len(re.findall(r"`|[{}]|=>|:=", question)),
        question.count("?"),
        sum(map(len, words)) / len(words) if words else 0,
        sum(char.isupper() for char in letters) / len(letters) if letters else 0,
    ]
    values.extend(
        len(re.findall(r"\b" + word + r"\b", question, re.I))
        for word in ("calculate", "compute", r"how\s+many", "explain", "why", "which")
    )
    return [float(value) for value in values]
