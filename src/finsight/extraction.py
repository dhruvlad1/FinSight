import re


def extract_guidance(text: str) -> list[str]:
    """Extract sentences that appear to contain management guidance."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    keywords = (
        "guidance",
        "expect",
        "expects",
        "forecast",
        "outlook",
        "target",
        "project",
        "projects",
        "anticipate",
        "anticipates",
    )

    return [
        sentence.strip()
        for sentence in sentences
        if any(keyword in sentence.lower() for keyword in keywords)
    ]
