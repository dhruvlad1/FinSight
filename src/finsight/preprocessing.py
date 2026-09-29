import re

import pandas as pd


def clean_text(text: str) -> str:
    """Normalize transcript text for downstream retrieval."""
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_transcripts(df: pd.DataFrame) -> pd.DataFrame:
    """Clean transcript text while preserving the original metadata."""
    result = df.copy()
    result["text"] = result["text"].apply(clean_text)
    result = result[result["text"].str.len() > 0].reset_index(drop=True)

    return result
