import json
from pathlib import Path

import pandas as pd


def load_transcripts(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    rows = []

    for record in records:
        for item in record["structured_content"]:
            rows.append(
                {
                    "company": record["company_name"],
                    "ticker": record["symbol"],
                    "quarter": record["quarter"],
                    "year": record["year"],
                    "date": record["date"],
                    "speaker": item["speaker"],
                    "text": item["text"],
                }
            )

    return pd.DataFrame(rows)
