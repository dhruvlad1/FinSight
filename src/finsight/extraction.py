import re


GUIDANCE_PATTERNS = (
    r"\bwe expect\b",
    r"\bwe anticipate\b",
    r"\bwe project\b",
    r"\bour expectation is\b",
    r"\bour expectations are\b",
    r"\bour outlook\b",
    r"\bour guidance\b",
    r"\bwe forecast\b",
    r"\bwe target\b",
    r"\bwe are targeting\b",
    r"\bwe plan to\b",
    r"\bwe're planning for\b",
    r"\bwe intend to\b",
    r"\bwe believe\b.*\bwill\b",
    r"\bexpected to\b",
    r"\bexpecting\b",
)

EXCLUDED_PATTERNS = (
    r"\bquestion-and-answer\b",
    r"\bq&a\b",
    r"\bjoining\b.*\bq&a\b",
    r"\bwill be followed by\b",
    r"\bwill be\b.*\bquestion\b",
)


def extract_guidance(text: str) -> list[str]:
    """Extract likely management guidance statements from transcript text."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    guidance = []

    for sentence in sentences:
        normalized = " ".join(sentence.split())
        lower = normalized.lower()

        if any(re.search(pattern, lower) for pattern in EXCLUDED_PATTERNS):
            continue

        if any(re.search(pattern, lower) for pattern in GUIDANCE_PATTERNS):
            guidance.append(normalized)

    return guidance