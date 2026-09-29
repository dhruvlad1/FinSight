from datasets import load_dataset
import json
from pathlib import Path

TARGET_COMPANIES = 5
QUARTERS_PER_COMPANY = 4

ds = load_dataset(
    "kurry/sp500_earnings_transcripts",
    split="train",
    streaming=True,
)

selected = {}
company_order = []

for row in ds:
    symbol = row["symbol"]

    if symbol not in selected:
        if len(company_order) >= TARGET_COMPANIES:
            continue
        selected[symbol] = []
        company_order.append(symbol)

    if len(selected[symbol]) < QUARTERS_PER_COMPANY:
        selected[symbol].append(row)

    if (
        len(company_order) == TARGET_COMPANIES
        and all(len(rows) >= QUARTERS_PER_COMPANY for rows in selected.values())
    ):
        break

records = [
    row
    for symbol in company_order
    for row in selected[symbol]
]

output = Path("data/raw/earnings_transcripts.json")
output.parent.mkdir(parents=True, exist_ok=True)

with output.open("w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"Saved {len(records)} transcripts.")
print("Companies:", company_order)
