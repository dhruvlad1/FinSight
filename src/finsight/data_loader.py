from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "company",
    "ticker",
    "quarter",
    "year",
    "date",
    "speaker",
    "speaker_type",
    "section",
    "text",
}


def load_transcripts(path: str | Path) -> pd.DataFrame:
    """Load earnings-call transcripts from a CSV file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Transcript dataset not found: {path}")

    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    return df


if __name__ == "__main__":
    print("FinSight data loader ready.")
