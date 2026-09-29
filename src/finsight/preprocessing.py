import pandas as pd


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = " ".join(text.split())
    return text.strip()


def preprocess_transcripts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].apply(clean_text)

    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    return df